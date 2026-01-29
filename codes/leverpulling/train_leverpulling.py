# train_leverpulling.py
import argparse
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from commnet import CommNet
from leverpulling_env import LeverPullingEnv
from independent_controller import IndependentController


def evaluate_greedy(model, env, num_trials=500, batch_size=64, device="cpu"):
    """
    Évalue la politique greedy (argmax) comme dans le papier:
    métrique = #levers distincts / m, moyenne sur 500 trials.
    """
    model.eval()
    total_reward = 0.0
    remaining = num_trials

    with torch.no_grad():
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            agent_ids = env.sample_agent_ids(current_batch, device=device)
            logits = model(agent_ids)
            actions = logits.argmax(dim=-1)
            rewards = env.compute_reward(actions)  # (B,)
            total_reward += rewards.sum().item()
            remaining -= current_batch

    model.train()
    return total_reward / float(num_trials)

@torch.no_grad()
def evaluate_sampled(model, env, num_trials=500, device="cpu"):
    model.eval()
    agent_ids = env.sample_agent_ids(num_trials, device=device)   # (T, M)
    logits = model(agent_ids)                                     # (T, M, A)
    dist = Categorical(logits=logits)
    actions = dist.sample()                                       # (T, M)
    rewards = env.compute_reward(actions)                         # (T,)
    return rewards.mean().item()

def evaluate_greedy_final(model, env, device, num_episodes=1000):
    model.eval()
    rewards = []

    with torch.no_grad():
        for _ in range(num_episodes):
            agent_ids = env.sample_agent_ids(1, device=device)  # batch=1
            logits = model(agent_ids)
            actions = logits.argmax(dim=-1)
            reward = env.compute_reward(actions)
            rewards.append(reward.item())

    return sum(rewards) / len(rewards)


def train_supervised(
    model,
    env,
    device="cpu",
    num_steps=50000,
    batch_size=64,
    lr=1e-3,
    print_every=500,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(1, num_steps + 1):
        agent_ids = env.sample_agent_ids(batch_size, device=device)
        logits = model(agent_ids)  # (B, M, num_levers)
        targets = env.optimal_actions(agent_ids)  # (B, M)

        loss = F.cross_entropy(
            logits.view(-1, env.num_levers), targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            greedy_score = evaluate_greedy(model, env, device=device)
            print(
                f"[Step {step:6d}] Loss: {loss.item():.4f}  "
                f"Greedy eval: {greedy_score:.3f}"
            )

def train_rl(
    model,
    env,
    device="cpu",
    num_steps=50000,
    batch_size=64,
    lr=1e-3,
    print_every=500,
    baseline_momentum=0.9,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    baseline = 0.0
    model.train()

    for step in range(1, num_steps + 1):
        agent_ids = env.sample_agent_ids(batch_size, device=device)

        logits = model(agent_ids) # (B, M, A)
        dist = Categorical(logits=logits)
        actions = dist.sample() # (B, M)
        log_probs = dist.log_prob(actions) # (B, M)

        rewards = env.compute_reward(actions) # (B,)

        # Baseline minimal (REINFORCE with baseline)
        # baseline EMA (standard)
        batch_mean = rewards.mean().item()
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * batch_mean
        adv = rewards - baseline
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8) # normalisation

        # Loss REINFORCE
        # loss = -(log_probs.sum(dim=1) * adv.detach()).mean()
        entropy = dist.entropy().sum(dim=1).mean()  # somme sur agents, moyenne batch
        loss = -(log_probs.sum(dim=1) * adv.detach()).mean() - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % print_every == 0:
            greedy_score = evaluate_greedy(model, env, device=device)
            print(
                f"[Step {step:6d}] loss={loss.item():.4f} "
                f"r_mean={rewards.mean().item():.3f} baseline={baseline:.3f} "
                f"greedy={greedy_score:.3f}"
            )


def main():
    # Parametres a rajouter pour l'execution dans le terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="supervised",
        choices=["RL", "supervised"],
        help='Mode d\'apprentissage: "RL" ou "supervised"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Ex: "cpu" ou "cuda"',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="commnet",
        choices=["commnet", "independant"],
        help='Modele: "commnet" ou "independant"',
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Hyperparamètres pour le lever pulling task
    num_agents_total = 500
    num_levers = 5
    group_size = 5

    env = LeverPullingEnv(
        num_agents_total=num_agents_total,
        num_levers=num_levers,
        group_size=group_size,
    )

    if args.model == "commnet":
        model = CommNet(
            num_agents_total=500,
            n_actions=5,
            hidden_dim=128,
            comm_steps=2,
            nonlin="relu",
        ).to(device)

    if args.model == "independant":
        model = IndependentController(
            obs_dim=500,
            n_actions=5,
            hidden_dim=128,
            comm_steps=2,
            use_h0=True,
            use_skip=True,
            nonlin="relu",
            obs_type="discrete",
        ).to(device)

    print("f in_features =", model.f[0].in_features if hasattr(model.f, "__getitem__") else "not_seq")

    if args.mode == "supervised":
        train_supervised(model, env, device=device, num_steps=50000, batch_size=64, lr=1e-3, print_every=500)

        final_score = evaluate_greedy_final(model, env, device=device, num_episodes=500)
        print(f"FINAL GREEDY SCORE: {final_score:.3f}")

    else:
        train_rl(model, env, device=device, num_steps=50000, batch_size=64, lr=1e-3, print_every=500)

        final_score = evaluate_greedy_final(model, env, device=device, num_episodes=1000)
        print(f"FINAL GREEDY SCORE: {final_score:.3f}")


if __name__ == "__main__":
    main()
