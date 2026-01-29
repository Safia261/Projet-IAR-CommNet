import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedController(nn.Module):
    """
    Contrôleur entièrement connecté où chaque agent partage ses observations avec tous les autres.
    Mais pas de communication itérative ni de flexibilité dans le nb d'agents.
    """

    def __init__(
        self,
        obs_dim: int,
        group_size: int,
        n_actions: int,
        hidden_dim: int = 128,
        nonlin: str = "relu",
        module: str = "mlp",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.group_size = group_size # = nombre d'agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.module = module

        if nonlin.lower() == "relu":
            self.act = nn.ReLU()
            act_fn = F.relu
        else:
            self.act = nn.Tanh()
            act_fn = torch.tanh

        self.nonlin = act_fn

        # ENCODER
        self.encoder = nn.Linear(obs_dim, hidden_dim) # peut etre pas besoin d'embedding ici, sion Linear car obs censées être continues

        self.f = self._create_module()

        # DECODER
        self.decoder = nn.Linear(hidden_dim, group_size * n_actions) # car il faut prédire une action pour tous les agents et "multiple output softmax heads" ensuite

        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.1)
        # à vérifier pour les 2 lignes suivantes
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.decoder.bias)   


    def _create_module(self):
        """
        Renvoie la couche avec le module f() choisi.
        """
        if self.module == "mlp":
            return nn.Sequential(nn.Linear(self.group_size * self.hidden_dim, self.hidden_dim), self.act)

        elif self.module == "rnn":
            return nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)

        elif self.module == "lstm":
            return nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)

        else:
            raise ValueError("Type du module incorrect. Doit être de type MLP, RNN ou LSTM.")


    def forward(self, obs):
        """
        """
        B, M, _ = obs.shape

        h = self.encoder(obs)  # h0 = (B, M, H)

        if self.module == "mlp":
            # concaténer les états cachés h0 des agents
            h = h.reshape(B, M * self.hidden_dim)
            h = self.f(h)  # (B, H)

        else:  # rnn / lstm
            out, _ = self.f(h)
            h = out[:, -1, :]  # dernier état caché après avoir vu tous les agents

        h = self.decoder(h)  # (B, M*n_actions)
        logits = h.reshape(B, M, self.n_actions)
        return logits
