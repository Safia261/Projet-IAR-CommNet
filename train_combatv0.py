import gym
import ma_gym       # Attention: only compatible with python versions < 3.12
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from models import ModelConfig, make_model

# ============================================================
# Useful functions for ma-gym Combat-v0
# ============================================================

def flatten_obs(obs_dict):
    """
    ma-gym retourne un dict {agent_id: obs}
    -> on transforme en array (N, obs_dim)
    """
    obs = []
    for agent_id in sorted(obs_dict.keys()):
        obs.append(obs_dict[agent_id])
    return np.array(obs, dtype=np.float32)


def flatten_rewards(reward_list):
    """
    reward_list: List[float] (une reward par agent)
    -> reward scalaire coopératif
    """
    return sum(reward_list)


# ============================================================
# Collect of one episode
# ============================================================

def obs_list_to_tensor(obs_list, device="cpu"):
    """
    obs_list: List[np.ndarray] de taille M
    -> tensor (1, M, obs_dim)
    """
    obs = torch.tensor(obs_list, dtype=torch.float32, device=device)
    return obs.unsqueeze(0)


def collect_episode(env, model, gamma=0.99, device="cpu"):
    log_probs = []
    rewards = [] # to compute win rate

    obs_list = env.reset()
    obs = obs_list_to_tensor(obs_list, device)
    n_agents = obs.shape[1]
    mask = full_mask(1, n_agents, device)

    done = False
    while not done:
        out = model(obs=obs, mask=mask)

        logits = out["action_logits"]
        dist = Categorical(logits=logits)

        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum()

        action_dict = {i: actions[0, i].item() for i in range(n_agents)}

        next_obs_list, reward_list, done_list, info = env.step(action_dict)

        reward = sum(reward_list)

        log_probs.append(log_prob)
        rewards.append(reward)

        done = all(done_list)
        obs = obs_list_to_tensor(next_obs_list, device)

    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    return log_probs, returns, sum(rewards)



# ============================================================
# Training
# ============================================================

def train(
    env,
    model,
    optimizer,
    epochs=150,
    updates_per_epoch=40,
    batch_size=16,
    gamma=0.99,
    device="cpu"
):
    model.train()
    win_rates = []

    for epoch in range(epochs):
        for _ in range(updates_per_epoch):
            all_log_probs = []
            all_returns = []

            for _ in range(batch_size):
                log_probs, returns, _ = collect_episode(
                    env, model, gamma, device
                )
                all_log_probs.extend(log_probs)
                all_returns.extend(returns)

            returns_tensor = torch.stack(all_returns)
            baseline = returns_tensor.mean()
            advantages = returns_tensor - baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # loss computation
            loss = 0.0
            for lp, adv in zip(all_log_probs, advantages):
                loss += -lp * adv
            loss = loss / len(advantages)

            optimizer.zero_grad()
            loss.backward() # loss backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        win_rate = test_policy(env, model, n_episodes=50, device=device)
        win_rates.append(win_rate)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Win-rate: {win_rate:.3f}"
        )

    return win_rates


# ============================================================
# Test
# ============================================================
def test_policy(env, model, n_episodes=100, device="cpu"):
    model.eval()
    wins = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            obs_list = env.reset()
            obs = obs_list_to_tensor(obs_list, device)
            n_agents = obs.shape[1]
            mask = full_mask(1, n_agents, device)

            done = False
            final_info = None

            while not done:
                out = model(obs=obs, mask=mask)
                logits = out["action_logits"]
                actions = torch.argmax(logits, dim=-1)

                action_dict = {i: actions[0, i].item() for i in range(n_agents)}

                next_obs_list, reward_list, done_list, info = env.step(action_dict)

                done = all(done_list)
                obs = obs_list_to_tensor(next_obs_list, device)
                final_info = info

            if final_info is not None and final_info.get("winner") == "blue":
                wins += 1

    model.train()
    return wins / n_episodes




# ============================================================
# Main
# ============================================================

def main():
    device = "cpu"

    env = gym.make("Combat-v0")
    n_agents = env.n_agents
    obs_dim = env.observation_space[0].shape[0]
    n_actions = env.action_space[0].n

    cfg = ModelConfig(
        obs_dim=obs_dim,          # dim observation per agent
        n_actions=n_actions,      # env.action_space[0].n
        n_agents=n_agents,        # env.n_agents
        hidden_dim=50, 
        module="mlp",
        k_steps=2,
        use_skip=True,
    )

    model = make_model("commnet", cfg).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    train(
        env,
        model,
        optimizer,
        epochs=70,
        updates_per_epoch=50,
        batch_size=16,
        device=device,
    )

    final_win_rate = test_policy(env, model, n_episodes=200, device=device)
    print(f"Win-rate final: {final_win_rate:.3f}")


def obs_list_to_tensor(obs_list, device):
    obs = torch.tensor(obs_list, dtype=torch.float32, device=device)
    return obs.unsqueeze(0)   # should be [1, N, obs_dim]


def full_mask(batch_size, n_agents, device):
    return torch.ones(batch_size, n_agents, device=device)


if __name__ == "__main__":
    main()