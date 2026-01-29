import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from combattask_env import CombatTask
# from commnet import CommNet
from models import ModelConfig, make_model


# def collect_episode(env, model, gamma=0.99):
#     obs = env.reset()
#     obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

#     log_probs = []
#     rewards = []

#     done = False
#     while not done:
#         logits = model(obs=obs)
#         dist = torch.distributions.Categorical(logits=logits)

#         actions = dist.sample()
#         log_prob = dist.log_prob(actions)  # (1, m)

#         actions_list = actions.squeeze(0).tolist()
#         next_obs, reward, done, info = env.step(actions_list)

#         log_probs.append(log_prob.sum())   # attention scalaire
#         rewards.append(reward)

#         obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

#     # returns
#     returns = []
#     G = 0.0
#     for r in reversed(rewards):
#         G = r + gamma * G
#         returns.insert(0, G)
#     returns = torch.tensor(returns, dtype=torch.float32)

#     return log_probs, returns, sum(rewards)



def collect_episode(env, model, gamma=0.99):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    n_agents = obs.shape[1]
    mask = torch.ones(1, n_agents)

    log_probs = []
    rewards = []

    done = False
    while not done:
        out = model(obs=obs, mask=mask)
        logits = out["action_logits"]

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum()

        actions_list = actions.squeeze(0).tolist()
        next_obs, reward, done, info = env.step(actions_list)

        log_probs.append(log_prob)
        rewards.append(reward)

        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    # returns
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    return log_probs, returns, sum(rewards)




def train_combattask(
    env,
    model,
    optimizer,
    gamma=0.99,
    epochs=150,
    updates_per_epoch=40,
    batch_size=16,
):
    model.train()
    total_episodes_seen = 0
    win_rates_history = []

    for epoch in range(epochs):
        epoch_return = 0.0

        for update in range(updates_per_epoch):

            all_log_probs = []
            all_returns = []
            batch_return = 0.0

            for _ in range(batch_size):
                log_probs, returns, ep_return = collect_episode(env, model, gamma)

                all_log_probs.extend(log_probs)
                all_returns.extend(returns)
                batch_return += ep_return
                total_episodes_seen += 1

            returns_tensor = torch.stack(all_returns)

            # baseline
            baseline = returns_tensor.mean()
            advantages = returns_tensor - baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # calcul de la loss
            loss = 0.0
            for log_p, adv in zip(all_log_probs, advantages):
                loss += -log_p * adv

            loss = loss / len(advantages)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_return += batch_return / batch_size

        avg_epoch_return = epoch_return / updates_per_epoch
        win_rate = test_policy(env, model, n_episodes=200)
        win_rates_history.append(win_rate)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Return moy: {avg_epoch_return:.2f} | "
            f"Episodes vus: {total_episodes_seen} | "
            f"Win-rate: {win_rate:.3f}"
        )

    return win_rates_history



# def test_policy(env, model, n_episodes=1000):
#     model.eval()
#     total_wins = 0

#     for _ in range(n_episodes):
#         obs = env.reset()
#         obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

#         done = False
#         final_result = None

#         while not done:
#             with torch.no_grad():
#                 logits = model(obs=obs)
#                 dist = torch.distributions.Categorical(logits=logits)
#                 actions = dist.sample()

#             actions_list = actions.squeeze(0).tolist()
#             next_obs, reward, done, info = env.step(actions_list)

#             if done:
#                 final_result = info["result"]

#             obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

#         if final_result == "win":
#             total_wins += 1

#     win_rate = total_wins / n_episodes
#     return win_rate


def test_policy(env, model, n_episodes=1000):
    model.eval()
    total_wins = 0

    for _ in range(n_episodes):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        n_agents = obs.shape[1]
        mask = torch.ones(1, n_agents)

        done = False
        final_result = None

        while not done:
            with torch.no_grad():
                out = model(obs=obs, mask=mask)
                logits = out["action_logits"]
                actions = torch.argmax(logits, dim=-1)

            actions_list = actions.squeeze(0).tolist()
            next_obs, reward, done, info = env.step(actions_list)

            if done:
                final_result = info["result"]

            obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        if final_result == "win":
            total_wins += 1

    return total_wins / n_episodes



def main():
    # 1) Environnement (sans rendu pour l'entraînement massif)
    env = CombatTask(render_mode=False)

    # 2) Modèle CommNet (hidden_dim=50 comme dans l’article, comm_steps=2)
    # pour l'ancienne version de CommNet
    # model = CommNet(
    #     num_agents_total=env.n_agents_per_team,
    #     n_actions=env.n_actions,
    #     hidden_dim=50,
    #     comm_steps=2,
    #     encoder_type="obs",
    #     obs_dim=env.obs_dim,
    #     module="mlp",
    # )

    cfg = ModelConfig(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        n_agents=env.n_agents_per_team,
        hidden_dim=50,
        module="mlp",
        k_steps=2,
        use_skip=True,
    )

    model = make_model("commnet", cfg)

    # 3) Optimiseur RMSProp (comme dans l’article)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    # 4) Entraînement
    win_rates_history = train_combattask(
        env,
        model,
        optimizer,
        gamma=0.99,
        epochs=150,
        updates_per_epoch=40,
        batch_size=16,
    )

    mean_win_rate = sum(win_rates_history) / len(win_rates_history)
    print(f"Taux de réussite moyen sur l'ensemble des epochs : {mean_win_rate:.3f}")

    # 5) Test final (pour évaluer l'entraînement)
    test_episodes = 1000
    win_rate = test_policy(env, model, n_episodes=test_episodes)
    print(f"Win-rate final sur {test_episodes} épisodes : {win_rate:.3f}")


if __name__ == "__main__":
    main()
