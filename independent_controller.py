import torch
import torch.nn as nn
import torch.nn.functional as F

class IndependentController(nn.Module):
    """
    Contrôleur indépendant pour chaque agent.

    Chaque agent observe uniquement son ID/ a ses propres obs, encode cette information
    via un embedding, puis prédit une distribution sur les actions.

    Pas de communication entre agents.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        nb_layers_mlp: int = 2,
        nonlin: str = "relu",
        module: str = "mlp",
        obs_type: str = "continuous", # "discrete" ou "continuous" 
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.nb_layers_mlp = nb_layers_mlp
        self.module = module

        if nonlin.lower() == "relu":
            self.act = nn.ReLU()
            act_fn = F.relu
        else:
            self.act = nn.Tanh()
            act_fn = torch.tanh

        self.nonlin = act_fn

        # ENCODER
        if obs_type == "continuous":
            self.encoder = nn.Linear(obs_dim, hidden_dim)
        else:
            self.encoder = nn.Embedding(obs_dim, hidden_dim)

        self.f = self._create_module()

        # DECODER
        self.decoder = nn.Linear(hidden_dim, n_actions)

        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.decoder.bias)


    def _create_module(self):
        """
        Renvoie la couche avec le module f() choisi.
        """
        if self.module == "mlp":
            if self.nb_layers_mlp < 1:
                raise ValueError("nb_layers_mlp doit être >= 1.")
            
            elif self.nb_layers_mlp == 1:
                return nn.Linear(self.hidden_dim, self.hidden_dim)
            
            elif self.nb_layers_mlp > 1:
                input_dim = self.hidden_dim

                layers = []
                in_dim = input_dim
                for i in range(self.nb_layers_mlp):
                    out_dim = self.hidden_dim
                    layers.append(nn.Linear(in_dim, out_dim))
                    if i < self.nb_layers_mlp - 1:
                        layers.append(self.act)
                    in_dim = out_dim

                return nn.Sequential(*layers)
        
        elif self.module == "rnn":
            return nn.RNN(self.hidden_dim, self.hidden_dim, batch_first = True)
        
        elif self.module == "lstm":
            return nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first = True)
        
        else:
            raise ValueError("Type du module incorrect. Doit être de type MLP, RNN ou LSTM.") 


    def forward(self, obs):
        """
        obs: (B, M) long tensor avec les obs (ID par ex) des agents dans chaque groupe

        Retourne:
            logits: (B, M, n_actions) scores non normalisés pour chaque action
        """
        if obs.dim() == 2: # pour lever pulling
            B, M = obs.shape
        else: # pour d'autres tâches avec obs continues
            B, M, _ = obs.shape
        
        # Encoder chaque agent indépendamment
        h = self.encoder(obs)  # (B, M, hidden_dim), h0 

        if self.module == "mlp":
            # Appliquer le module f indépendamment à chaque agent
            h = h.reshape(B * M, -1)  # (B*M, hidden_dim)
            h = self.f(h)  # (B*M, hidden_dim)
            h = h.reshape(B, M, -1)  # (B, M, hidden_dim)
        else:  # rnn / lstm
            out, _ = self.f(h)
            h = out  # (B, M, hidden_dim)

        logits = self.decoder(h)  # (B, M, n_actions)
        return logits