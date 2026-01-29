from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn

from traffic_junction_env import TrafficJunctionConfig, TrafficJunctionEnv


@dataclass
class TrainConfig:
    # Paper
    batch_episodes: int = 288
    epochs: int = 300
    updates_per_epoch: int = 100

    # Optim (paper: RMSProp lr=0.003)
    lr: float = 0.003
    rms_alpha: float = 0.97
    rms_eps: float = 1e-6
    grad_clip: float = 5.0

    # Appendix A: alpha = 0.03 for balancing reward and baseline objectives.
    alpha_baseline: float = 0.03

    entropy_coef: float = 0.0

    # Appendix C: curriculum on p_arrive (0.05 -> 0.2).
    use_curriculum: bool = True
    p_arrive_start: float = 0.05
    p_arrive_end: float = 0.20

    device: str = "cpu"
    seed: int = 1


def compute_returns_undiscounted(rewards, dones):
    """Undiscounted return G_t = sum_{i=t}^T r_i (Appendix A)."""
    # rewards: [E,T] dones: [E,T]
    E, T = rewards.shape
    returns = torch.zeros_like(rewards)

    for e in range(E):
        g = 0.0
        for t in reversed(range(T)):
            g = g + rewards[e, t]
            returns[e, t] = g

            if dones[e, t] > 0.5:
                g = 0.0

    return returns


def curriculum_p_arrive(epoch_idx, total_epochs, p0, p1):
    """Appendix C schedule:
      - first 100 epochs: p_arrive = 0.05
      - next 100 epochs: linear increase to 0.2
      - last 100 epochs: p_arrive = 0.2

    If total_epochs != 300, we scale the 1/3,1/3,1/3 segments proportionally.
    """
    if total_epochs <= 0:
        return p1

    if total_epochs == 300:
        if epoch_idx < 100:
            return p0
        if epoch_idx < 200:
            # epoch 100 -> p0, epoch 199 -> p1 (100 points)
            denom = 99.0
            frac = (epoch_idx - 100) / denom
            return p0 + frac * (p1 - p0)
        return p1

    a = total_epochs // 3
    b = 2 * total_epochs // 3
    if epoch_idx < a:
        return p0
    if epoch_idx < b:
        # epoch a -> p0, epoch (b-1) -> p1
        denom = float(max(b - a - 1, 1))
        frac = (epoch_idx - a) / denom
        return p0 + frac * (p1 - p0)
    return p1


@torch.no_grad()
def evaluate_failure_rate(
    model: nn.Module,
    env_cfg: TrafficJunctionConfig,
    episodes: int,
    device: torch.device,
    seed: int = 0,
    greedy: bool = True,
    n_comm_symbols: int = 10,
):
    """Failure rate = fraction of episodes with >=1 collision (paper)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()

    fails = 0
    rets = []
    for ep in range(episodes):
        env = TrafficJunctionEnv(env_cfg, seed=seed + ep * 9973)
        obs = env.reset()

        state = model.init_state(1, env_cfg.nmax, device) if hasattr(model, "init_state") else None

        done = False
        ep_ret = 0.0
        ever_collide = False

        while not done:
            obs_t = torch.tensor(obs["obs"], device=device).unsqueeze(0)
            mask_t = torch.tensor(obs["mask_active"], device=device).unsqueeze(0)

            if model.__class__.__name__ == "DiscreteCommPolicy":
                out = model(obs_t, mask_t, comm_mode="greedy" if greedy else "sample", state=state)
            else:
                out = model(obs_t, mask_t, state=state)

            logits = out["action_logits"]
            if greedy:
                act = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()

            state = out.get("next_state", None)

            obs, r, done, info = env.step(act.squeeze(0).cpu().numpy().astype(np.int64))
            ep_ret += float(r)
            if info.get("collision_cells", 0) > 0:
                ever_collide = True

        if ever_collide:
            fails += 1
        rets.append(ep_ret)

    return fails / float(episodes), float(np.mean(rets))


def rollout_batch(
    model: nn.Module,
    env_cfg: TrafficJunctionConfig,
    batch_episodes: int,
    device: torch.device,
    seed: int,
    n_comm_symbols: int = 10,
):
    """Vectorized rollout over a batch of environments.

    Returns tensors needed for Appendix A update:
      mask
      logp
      comm_logp:
      value
      entropy
      reward
      done
    """
    model.train()

    E = batch_episodes
    N = env_cfg.nmax
    T = env_cfg.max_steps

    envs = [TrafficJunctionEnv(env_cfg, seed=seed + e * 1337) for e in range(E)]
    obs_list = [env.reset() for env in envs]

    state = model.init_state(E, N, device) if hasattr(model, "init_state") else None

    masks = []
    logps = []
    comm_logps = []
    values_list = []
    ents = []
    rewards_list = []
    dones_list = []

    for t in range(T):
        obs_np = np.stack([o["obs"] for o in obs_list], axis=0) # [E,N,D]
        mask_np = np.stack([o["mask_active"] for o in obs_list], axis=0) # [E,N]

        obs_t = torch.tensor(obs_np, device=device)
        mask_t = torch.tensor(mask_np, device=device)

        if model.__class__.__name__ == "DiscreteCommPolicy":
            out = model(obs_t, mask_t, comm_mode="sample", state=state)
            comm_logp = out["comm_logp"] # [E,N]

        else:
            out = model(obs_t, mask_t, state=state)
            comm_logp = torch.zeros(E, N, device=device)

        logits = out["action_logits"] # [E,N,A]
        values = out["baseline"] # [E,N]
        state = out.get("next_state", None)

        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample() # [E,N]
        logp = dist.log_prob(act) # [E,N]
        ent = dist.entropy() # [E,N]

        rewards = np.zeros((E,), dtype=np.float32)
        dones = np.zeros((E,), dtype=np.float32)
        next_obs_list = []

        for e, env in enumerate(envs):
            o2, r, d, info = env.step(act[e].detach().cpu().numpy().astype(np.int64))
            next_obs_list.append(o2)
            rewards[e] = float(r)
            dones[e] = float(d)

        obs_list = next_obs_list

        masks.append(mask_t)
        logps.append(logp * mask_t)
        comm_logps.append(comm_logp * mask_t)
        values_list.append(values * mask_t)
        ents.append(ent * mask_t)
        rewards_list.append(torch.tensor(rewards, device=device))
        dones_list.append(torch.tensor(dones, device=device))

    mask = torch.stack(masks, dim=1) # [E,T,N]
    logp = torch.stack(logps, dim=1) # [E,T,N]
    comm_logp = torch.stack(comm_logps, dim=1) # [E,T,N]
    value = torch.stack(values_list, dim=1) # [E,T,N]
    entropy = torch.stack(ents, dim=1) # [E,T,N]
    reward = torch.stack(rewards_list, dim=1) # [E,T]
    done = torch.stack(dones_list, dim=1) # [E,T]

    return {"mask": mask, "logp": logp, "comm_logp": comm_logp, "value": value, "entropy": entropy, "reward": reward, "done": done}


def train_reinforce(model: nn.Module, env_cfg: TrafficJunctionConfig, cfg: TrainConfig, n_comm_symbols = 10):
    device = torch.device(cfg.device)
    model.to(device)
    optim = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=cfg.rms_alpha, eps=cfg.rms_eps)

    history = {"loss": [], "mean_return": [], "fail_rate": [], "p_arrive": []}

    global_step = 0
    for epoch in range(cfg.epochs):

        if cfg.use_curriculum:
            env_cfg.p_arrive = float(curriculum_p_arrive(epoch, cfg.epochs, cfg.p_arrive_start, cfg.p_arrive_end))

        for _ in range(cfg.updates_per_epoch):
            batch = rollout_batch(
                model=model,
                env_cfg=env_cfg,
                batch_episodes=cfg.batch_episodes,
                device=device,
                seed=cfg.seed + global_step * 10007,
                n_comm_symbols=n_comm_symbols,
            )

            rewards = batch["reward"] # [E,T]
            dones = batch["done"] # [E,T]
            mask = batch["mask"] # [E,T,N]
            values = batch["value"] # [E,T,N]
            logp = batch["logp"] # [E,T,N]
            comm_logp = batch["comm_logp"] # [E,T,N]

            returns = compute_returns_undiscounted(rewards, dones) # [E,T]
            returns = returns.unsqueeze(-1).expand_as(values) # [E,T,N]

            # Appendix A, Eq (7)
            advantage = (returns - values) * mask
            logp_total = (logp + comm_logp) * mask

            denom = mask.sum().clamp_min(1.0)
            policy_loss = -(logp_total * advantage.detach()).sum() / denom
            baseline_loss = cfg.alpha_baseline * (advantage ** 2).sum() / denom
            loss = policy_loss + baseline_loss

            if cfg.entropy_coef and cfg.entropy_coef > 0.0:
                entropy_mean = (batch["entropy"] * mask).sum() / denom
                loss = loss - cfg.entropy_coef * entropy_mean

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            with torch.no_grad():
                mean_ret = float(returns[:, 0, 0].mean().item())
                fail = (rewards.min(dim=1).values <= env_cfg.r_collision).float().mean().item()

            history["loss"].append(float(loss.item()))
            history["mean_return"].append(mean_ret)
            history["fail_rate"].append(float(fail))
            history["p_arrive"].append(float(env_cfg.p_arrive))

            global_step += 1

        # print(f"Epoch {epoch}: done")

    return history
