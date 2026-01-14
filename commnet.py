# commnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CommNet(nn.Module):
    """
    CommNet unique, utilisable pour:
      - LeverPulling: encoder_type="id" (par défaut) + forward(agent_ids)
      - TrafficJunction / Combat: encoder_type="obs" + forward(obs=...)

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
        module="mlp",

        encoder_type: str = "id",  # "id" (leverpulling) ou "obs" (traffic/combat)
        obs_dim: int | None = None,
    ):
        super().__init__()

        self.num_agents_total = num_agents_total
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.comm_steps = comm_steps
        self.use_h0 = use_h0
        self.use_skip = use_skip
        self.module = module
        self.encoder_type = encoder_type
        self.obs_dim = obs_dim

        if nonlin.lower() == "relu":
            self.act = nn.ReLU()
            act_fn = F.relu
        else:
            self.act = nn.Tanh()
            act_fn = torch.tanh
        self.nonlin = act_fn

        # ENCODER
        if self.encoder_type == "id":
            # LeverPulling: agent_ids -> embedding
            self.encoder = nn.Embedding(num_agents_total, hidden_dim)
        elif self.encoder_type == "obs":
            # Traffic/Combat: obs features -> linear
            if obs_dim is None:
                raise ValueError('obs_dim doit être fourni si encoder_type="obs".')
            self.encoder = nn.Linear(obs_dim, hidden_dim)
        else:
            raise ValueError('encoder_type doit être "id" ou "obs".')

        # Module f (MLP / RNN / LSTM)
        self.f = self._create_module()

        # DECODER
        self.decoder = nn.Linear(hidden_dim, n_actions)

        self._init_weights()

    def _init_weights(self):
        # Init compatible avec les deux encoders
        if isinstance(self.encoder, nn.Embedding):
            nn.init.normal_(self.encoder.weight, mean=0.0, std=0.1)

    def _create_module(self):
        """
        Renvoie la couche avec le module f() choisi.
        IMPORTANT: input_dim dépend de use_h0 (sinon mismatch pour RNN/LSTM).
        """
        input_dim = self.hidden_dim * (3 if self.use_h0 else 2)

        if self.module == "mlp":
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
            return nn.RNN(input_dim, self.hidden_dim, batch_first=True)

        elif self.module == "lstm":
            return nn.LSTM(input_dim, self.hidden_dim, batch_first=True)

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

    def forward(self, agent_ids=None, obs=None):
        """
        LeverPulling:
            logits = model(agent_ids)  # agent_ids: (B, M) long

        TrafficJunction / Combat:
            logits = model(obs=obs)    # obs: (B, M, obs_dim) float

        Sortie:
            logits : (B, M, n_actions)
        """
        # --- Encoder ---
        if self.encoder_type == "id":
            if agent_ids is None:
                raise ValueError('encoder_type="id": il faut fournir agent_ids.')
            h0 = self.encoder(agent_ids)  # (B,M,H)

        elif self.encoder_type == "obs":
            if obs is None:
                raise ValueError('encoder_type="obs": il faut fournir obs=...')
            h0 = self.encoder(obs)        # (B,M,H)

        else:
            raise RuntimeError("encoder_type inconnu (devrait être géré au __init__).")

        h = h0.clone()

        # --- Communication K steps ---
        for _ in range(self.comm_steps):
            c = self._compute_communication(h)

            if self.use_h0:
                x = torch.cat([h, c, h0], dim=-1)  # (B,M,3H)
            else:
                x = torch.cat([h, c], dim=-1)      # (B,M,2H)

            delta = self.f(x)
            # Pour RNN/LSTM, PyTorch renvoie un tuple (output, hidden)
            if isinstance(delta, tuple):
                delta = delta[0]  # output: (B,M,H)

            if self.use_skip:
                h = h + delta
            else:
                h = delta

        # --- Decode ---
        logits = self.decoder(h)
        return logits
