# leverpulling_env.py
import torch


class LeverPullingEnv:
    """
    Environnement pour la tâche de lever-pulling:

    - m leviers
    - N agents au total
    - À chaque round, on échantillonne m agents distincts parmi N
    - Chaque agent choisit un levier
    - Récompense = (# levers distincts) / m, partagee par tous les agents du round
    - Trier les IDs des agents et assigner le levier en fonction du rang (0, ..., m-1) dans ce tri.
    """

    def __init__(self, num_agents_total=500, num_levers=5, group_size=5):
        assert group_size == num_levers, "group_size = num_levers = 5"
        self.num_agents_total = num_agents_total
        self.num_levers = num_levers
        self.group_size = group_size

    def sample_agent_ids(self, batch_size, device=None):
        """
        Echantillonne pour chaque élément du batch un groupe de group_size
        agents distincts parmi N.

        Retourne:
            agent_ids: (B, M) long tensor
        """
        if device is None:
            device = "cpu"

        B = batch_size
        M = self.group_size
        agent_ids = torch.empty(B, M, dtype=torch.long, device=device)

        # Boucle batch_size (64) * 500 agents -> ok
        for b in range(B):
            perm = torch.randperm(self.num_agents_total, device=device)
            agent_ids[b] = perm[:M]

        return agent_ids

    def compute_reward(self, actions):
        """
        actions: (B, M) avec valeurs dans [0, num_levers-1]

        Retourne:
            reward: (B,) reward scalaire par round = (#levers distincts) / m
        """
        B, M = actions.shape
        rewards = torch.empty(B, device=actions.device, dtype=torch.float32)
        for b in range(B):
            num_distinct = actions[b].unique().numel()
            rewards[b] = num_distinct / float(self.num_levers)
        return rewards

    def optimal_actions(self, agent_ids):
        """
        Supervision oracle du papier:
        - On trie les IDs des agents dans le groupe
        - On affecte les leviers 0..m-1 selon l'ordre trié (rang dans le groupe)

        agent_ids: (B, M)

        Retourne:
            targets: (B, M) avec valeurs dans [0, num_levers-1]
        """
        B, M = agent_ids.shape
        device = agent_ids.device

        # tri par ligne
        sorted_ids, indices = torch.sort(agent_ids, dim=1)  # indices: positions dans l'original

        # ranks[b, original_pos] = rang dans le tri
        ranks = torch.empty_like(indices)
        arange = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        ranks.scatter_(1, indices, arange)

        # leviers = rang mod m (ici m = M)
        targets = ranks % self.num_levers
        return targets
