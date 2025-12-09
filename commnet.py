# Définition du modèle CommNet (Communication Neural Network)

import torch
from torch import nn

class CommNet(nn.Module):
    """
    Classe de CommNet.
    param = {
        "module" = "mlp",        # mlp / rnn / lstm (module f())
        "nonlin" = "tanh",       # tanh / relu (non-linéarité sigma)
        "hidden_dim" = 50,       # dimension de la couche cachée
        "input_dim" = 10,        # dimension de la couche d'entrée
        # "output_dim" = 10,     # dimension de la couche de sortie
        "comm_steps" = 2,        # pas de communication (K dans l'article)
        "nagents" = 10,          # nombre d'agents
        "nactions" = 4           # nombre d'actions (car actions discrètes)

        # ajouter (ou supprimer) par la suite les paramètres nécessaires
    }

    
    """
    def __init__(self, param):
        super(CommNet, self).__init__()

        self.param = param
        self.module = param["module"] # pas plutot self.param?
        self.nonlin = param["nonlin"]
        self.nagents = param["nagents"]
        self.nactions = param["nactions"]
        self.input_dim = param["input_dim"]
        # self.output_dim = param["output_dim"] # = nactions
        self.hidden_dim = param["hidden_dim"]
        self.comm_steps = param["comm_steps"]

        # Couche d'entrée / d'encodage
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)

        # Couches cachées
        self.hlayers = nn.ModuleList([self._create_module() for _ in range (self.comm_steps)])

        # Couche de sortie / de décodage
        self.decoder = nn.Linear(self.hidden_dim, self.nactions)

        if self.nonlin == "tanh":
            self.activation = nn.Tanh()
        elif self.nonlin == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Non-linéarité non conforme.")
    

    def _create_module(self):
        """
        Renvoie la couche avec le module f() choisi.
        """
        if self.module == "mlp":
            return nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            # multiplication par 2 car 2 vecteurs h et c en input par agent
        elif self.module == "rnn":
            return nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first = True)
        elif self.module == "lstm":
            return nn.LSTM(self.hidden_dim * 2, self.hidden_dim, batch_first = True)
        else:
            raise ValueError("Type du module incorrect. Doit être de type MLP, RNN ou LSTM.") 
    

    def forward(self, s):
        # voir plus tard la forme du tenseur s selon la suite du projet (s = concaténation des états initiaux s_j de chaque agent)
        # s de la forme (batch_size, nagents, input_dim) ?

        # Encodage des états s en états cachés h
        h = self.activation(self.encoder(s))

        # mask pour calculer c (moyenne des h des agents sauf celui de l'agent courant)
        mask = torch.ones(self.nagents, self.nagents, device=s.device) - torch.eye(self.nagents)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1) # de forme (1, nagents, nagents, 1)
        # risque de ne pas marcher si le nombre d'agents est dynamique

        # Passage par les couches cachées
        for k in range (self.comm_steps):
            # màj du vecteur de communication c (de la forme (batch, nagents, hidden_dim) ?)
            c = (h.unsqueeze(1) * mask_expanded).sum(dim = 2)/ (self.nagents - 1)  # et s(il y avait qu'1 seul agent?)
            # avec h.unsqueeze de forme (batch, 1, nagents, hidden_dim)

            # concaténation de h et c pour appliquer la non-linéarité
            hc_concat = torch.cat((h, c), dim = -1)

            # application de la non-linéarité
            if self.module == "mlp":
                h = self.activation(self.hlayers[k](hc_concat))
            elif self.module == "rnn" or self.module == "lstm":
                output, _ = self.hlayers[k](hc_concat)
                h = self.activation(output)
            
        logits = self.decoder(h) # logits = sorties de la dernière couche mais pas encore les actions -> softmax à appliquer
        return logits