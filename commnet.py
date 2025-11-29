# Définition du modèle CommNet (Communication Neural Network)

import torch
from torch import nn

class CommNet(nn.Module):
    """
    Classe de CommNet.
    param = {
        "module" = "mlp"        # mlp / rnn / lstm (module f())
        "nonlin" = "tanh"       # tanh / relu (non-linéarité sigma)
        "hidden_dim" = 50       # dimension de la couche cachée
        "input_dim" = 10        # dimension de la couche d'entrée
        "output_dim" = 10       # dimension de la couche de sortie
        "comm_steps" = 2        # pas de communication (K dans l'article)
        "nagents" = 10          # nombre d'agents
        "nactions" = 4          # nombre d'actions (car actions discrètes)

        # ajouter (ou supprimer) par la suite les paramètres nécessaires
    }

    
    """
    def __init__(self, param):
        super(CommNet, self).__init__()

        self.param = param
        self.module = param["module"]
        self.nonlin = param["nonlin"]
        self.nagents = param["nagents"]
        self.nactions = param["nactions"]
        self.input_dim = param["input_dim"]
        self.output_dim = param["output_dim"]
        self.hidden_dim = param["hidden_dim"]
        self.comm_steps = param["comm_steps"]

        # Couche d'entrée / d'encodage
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)

        # Couches cachées
        self.hlayers = nn.ModuleList([self._create_module()] for k in range (self.comm_steps))

        # Couche de sortie / de décodage
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)
    

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

      
    def _nonlin(self, dim_in, dim_out):
        """
        Applique la non-linéairité (sigma) choisie (correspondant à la fonction d'activation).
        """
        if self.nonlin == "tanh":
            return nn.Tanh()
        elif self.nonlin == "relu":
            return nn.ReLu()
        else:
            raise ValueError("Non-linéarité non conforme.")
    

    def forward(self, x):
        # voir plus tard la forme du tenseur x selon la suite du projet (x = concaténation des états initiaux s)

        # Encodage des états s en états cachés h
        h = self._nonlin(self.encoder(x))

        # Passage par les couches cachées
        for k in range (self.comm_steps):
            # màj du vecteur de communication c
            c = h.mean()

            # concaténation de h et c pour appliquer la non-linéarité
            hc_concat = torch.cat((h, c), dim = -1)

            # application de la non-linéarité
            if self.module == "mlp":
                h = self._nonlin(self.hlayer[k](hc_concat))
            elif self.module == "rnn" or self.module == "lstm":
                output, _ = self.hlayer[k](hc_concat)
                h = self._nonlin(output)
            
        logits = self.decoder(h) # logits = sorties de la dernière couche mais pas encore les actions -> softmax à appliquer
        return logits
