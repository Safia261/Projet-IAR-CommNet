# commnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CommNet(nn.Module):
    """
    Modèle CommNet générique :
    - Prend en entrée des IDs d'agents (optionnellement n'importe quels features si tu changes l'encodeur)
    - Encode chaque agent en h0
    - Applique K étapes de communication
    - Renvoie les logits d'action

    Utilisable pour lever-pulling, mais facilement extensible à d'autres environnements.
    """

    def __init__(
        self,
        num_agents_total: int,
        n_actions: int,
        hidden_dim: int = 128,
        comm_steps: int = 2,
        nonlin: str = "relu",
        use_h0: bool = True,
        use_skip: bool = True,
        module = "mlp",
    ):
        super().__init__()

        # ou peut etre revenir a la version d'avant?
        self.num_agents_total = num_agents_total
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.comm_steps = comm_steps
        self.use_h0 = use_h0
        self.use_skip = use_skip
        self.module = module

        if nonlin.lower() == "relu":
            self.act = nn.ReLU()
            act_fn = F.relu
        else:
            self.act = nn.Tanh()
            act_fn = torch.tanh

        self.nonlin = act_fn

        # ENCODER
        self.encoder = nn.Embedding(num_agents_total, hidden_dim)

        self.f = self._create_module()

        # DECODER
        self.decoder = nn.Linear(hidden_dim, n_actions)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.1)

    def _create_module(self):
        """
        Renvoie la couche avec le module f() choisi.
        """
        if self.module == "mlp":
            input_dim = self.hidden_dim * (3 if self.use_h0 else 2)

            layers = []
            in_dim = input_dim
            for i in range(2):
                out_dim = self.hidden_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < 2 - 1:
                    layers.append(self.act)
                in_dim = out_dim

            return nn.Sequential(*layers)

        elif self.module == "rnn":
            return nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first = True)
        
        elif self.module == "lstm":
            return nn.LSTM(self.hidden_dim * 2, self.hidden_dim, batch_first = True)
        
        else:
            raise ValueError("Type du module incorrect. Doit être de type MLP, RNN ou LSTM.") 


    @staticmethod
    def _compute_communication(h):
        """
        h : (B, M, H)
        return c : moyenne des h_k pour k != j
        """
        B, M, H = h.shape
        if M == 1:
            return torch.zeros_like(h)
        sum_all = h.sum(dim=1, keepdim=True)
        c = (sum_all - h) / (M - 1)
        return c

    def forward(self, agent_ids):
        """
        agent_ids : (B, M) indices des agents choisis pour ce round.

        Sortie :
            logits : (B, M, n_actions)
        """

        # 1) Encode les IDs -> h0
        h0 = self.encoder(agent_ids)  # (B, M, H)
        h = h0.clone()

        # 2) K étapes de communication
        for _ in range(self.comm_steps):
            c = self._compute_communication(h)

            if self.use_h0:
                x = torch.cat([h, c, h0], dim=-1)
            else:
                x = torch.cat([h, c], dim=-1)

            # a verifier
            delta = self.f(x)

            if self.use_skip:
                h = h + delta
            else:
                h = delta

        # 3) Décodage vers les actions
        logits = self.decoder(h)
        return logits
