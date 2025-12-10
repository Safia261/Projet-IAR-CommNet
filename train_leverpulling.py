# train_leverpulling.py
import argparse
import torch
import torch.nn.functional as F
from torch import optim

from commnet import CommNet
from leverpulling_env import LeverPullingEnv


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

def main():
    # Parametres a rajouter pour l'execution dans le terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="supervised",
        choices=["reinforce", "supervised"],
        help='Mode d\'apprentissage: "reinforce" ou "supervised"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Ex: "cpu" ou "cuda"',
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

    model = CommNet(
        num_agents_total=500,
        n_actions=5,
        hidden_dim=128,
        comm_steps=2,
        nonlin="relu",
    ).to(device)


    if args.mode == "supervised":
        train_supervised(
            model,
            env,
            device=device,
            num_steps=50000,
            batch_size=64,
            lr=1e-3,
            print_every=500,
        )
    else:
        print("a implementer !")


if __name__ == "__main__":
    main()
