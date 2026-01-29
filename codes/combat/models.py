from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    obs_dim: int
    hidden_dim: int = 50
    n_actions: int = 2
    n_agents: int = 10

    # "module" refers to the paper's f(.) choice for multi-turn games.
    # - mlp  : feedforward CommNet with K communication steps (K=2 in the paper)
    # - rnn  : RNN cell module used inside CommNet across communication steps (shared parameters)
    # - lstm : LSTM cell module used inside CommNet across communication steps (shared parameters)
    module: str = "mlp"  # "mlp" | "rnn" | "lstm"

    # number of communication steps (K). The paper uses K=2 for Traffic Junction.
    k_steps: int = 2

    # Skip connections (paper: MLP modules use skip connections).
    use_skip: bool = True

    # Communication dimension. In the CommNet paper, c and h have same dimensionality.
    comm_dim: int = 50

    # Discrete comm.
    n_comm_symbols: int = 10


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------
class MLPBlock(nn.Module):
    """Single-layer MLP block (paper: 'single layer network').

    The paper's experiments for Traffic Junction indicate using a single-layer network
    for the feedforward MLP module f(.). Non-linearity is applied by the caller
    (CommNet uses tanh in the paper figures/description).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class AgentCore(nn.Module):
    """Per-agent core used by the Independent baseline.

    For module='rnn'/'lstm', we keep temporal recurrence across environment steps
    (this is the usual interpretation for an independent controller baseline).
    """
    def __init__(self, obs_dim: int, hidden_dim: int, module: str):
        super().__init__()
        self.module = module
        if module == "mlp":
            self.net = MLPBlock(obs_dim, hidden_dim)
        elif module == "rnn":
            self.rnn = nn.RNNCell(obs_dim, hidden_dim)
        elif module == "lstm":
            self.lstm = nn.LSTMCell(obs_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown module: {module}")

    def init_state(self, batch: int, n_agents: int, device: torch.device):
        if self.module == "mlp":
            return None
        if self.module == "rnn":
            return torch.zeros(batch, n_agents, self.rnn.hidden_size, device=device)
        if self.module == "lstm":
            h = torch.zeros(batch, n_agents, self.lstm.hidden_size, device=device)
            c = torch.zeros(batch, n_agents, self.lstm.hidden_size, device=device)
            return (h, c)
        raise RuntimeError

    def forward(self, x: torch.Tensor, state=None):
        # x: [B,N,obs_dim]
        B, N, _ = x.shape
        if self.module == "mlp":
            h = torch.tanh(self.net(x))
            return h, None

        if self.module == "rnn":
            if state is None:
                state = torch.zeros(B, N, self.rnn.hidden_size, device=x.device)
            x2 = x.reshape(B * N, -1)
            h0 = state.reshape(B * N, -1)
            h1 = self.rnn(x2, h0).reshape(B, N, -1)
            return h1, h1

        if self.module == "lstm":
            if state is None:
                h0 = torch.zeros(B, N, self.lstm.hidden_size, device=x.device)
                c0 = torch.zeros(B, N, self.lstm.hidden_size, device=x.device)
            else:
                h0, c0 = state
            x2 = x.reshape(B * N, -1)
            h0f = h0.reshape(B * N, -1)
            c0f = c0.reshape(B * N, -1)
            h1, c1 = self.lstm(x2, (h0f, c0f))
            h1 = h1.reshape(B, N, -1)
            c1 = c1.reshape(B, N, -1)
            return h1, (h1, c1)

        raise RuntimeError


# ---------------------------------------------------------------------
# Policies / Controllers
# ---------------------------------------------------------------------
class IndependentPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.core = AgentCore(cfg.obs_dim, cfg.hidden_dim, cfg.module)
        self.pi = nn.Linear(cfg.hidden_dim, cfg.n_actions)
        self.v = nn.Linear(cfg.hidden_dim, 1)

    def init_state(self, batch: int, n_agents: int, device: torch.device):
        return self.core.init_state(batch, n_agents, device)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, state=None) -> Dict[str, torch.Tensor]:
        # obs: [B,N,D] mask: [B,N]
        h, next_state = self.core(obs, state)
        logits = self.pi(h)
        baseline = self.v(h).squeeze(-1)
        logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        baseline = baseline * mask
        return {"action_logits": logits, "baseline": baseline, "next_state": next_state}


class FullyConnectedPolicy(nn.Module):
    """Baseline where all agent observations are concatenated and processed jointly.

    This is NOT modular or permutation-invariant (paper's fully-connected baseline).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.obs_dim, cfg.hidden_dim)

        flat_dim = cfg.n_agents * cfg.hidden_dim
        if cfg.module == "mlp":
            self.core = MLPBlock(flat_dim, cfg.hidden_dim)
        elif cfg.module == "rnn":
            self.core = nn.RNNCell(flat_dim, cfg.hidden_dim)
        elif cfg.module == "lstm":
            self.core = nn.LSTMCell(flat_dim, cfg.hidden_dim)
        else:
            raise ValueError(cfg.module)

        self.pi = nn.Linear(cfg.hidden_dim, cfg.n_agents * cfg.n_actions)
        self.v = nn.Linear(cfg.hidden_dim, cfg.n_agents)

    def init_state(self, batch: int, n_agents: int, device: torch.device):
        if self.cfg.module == "mlp":
            return None
        if self.cfg.module == "rnn":
            return torch.zeros(batch, self.cfg.hidden_dim, device=device)
        if self.cfg.module == "lstm":
            h = torch.zeros(batch, self.cfg.hidden_dim, device=device)
            c = torch.zeros(batch, self.cfg.hidden_dim, device=device)
            return (h, c)
        raise RuntimeError

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, state=None) -> Dict[str, torch.Tensor]:
        B, N, _ = obs.shape
        emb = torch.tanh(self.embed(obs))  # [B,N,H]
        flat = emb.reshape(B, N * self.cfg.hidden_dim)

        if self.cfg.module == "mlp":
            g = torch.tanh(self.core(flat))
            next_state = None
        elif self.cfg.module == "rnn":
            if state is None:
                state = torch.zeros(B, self.cfg.hidden_dim, device=obs.device)
            g = self.core(flat, state)
            next_state = g
        else:  # lstm
            if state is None:
                h0 = torch.zeros(B, self.cfg.hidden_dim, device=obs.device)
                c0 = torch.zeros(B, self.cfg.hidden_dim, device=obs.device)
            else:
                h0, c0 = state
            h1, c1 = self.core(flat, (h0, c0))
            g, next_state = h1, (h1, c1)

        logits = self.pi(g).reshape(B, N, self.cfg.n_actions)
        baseline = self.v(g).reshape(B, N)
        logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        baseline = baseline * mask
        return {"action_logits": logits, "baseline": baseline, "next_state": next_state}


class DiscreteCommPolicy(nn.Module):
    """Discrete communication baseline (paper Section 4.1 Discrete communication).

    Agents output discrete symbols w_j^i at each communication step i.
    The receiver gets a bag-of-words style OR aggregation.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.obs_dim, cfg.hidden_dim)

        # f-module:
        # - MLP: separate parameters per communication step i (paper: f^i)
        # - RNN/LSTM: shared parameters across steps (cell)
        if cfg.module == "mlp":
            in_dim = cfg.hidden_dim + cfg.n_comm_symbols
            if cfg.use_skip:
                in_dim += cfg.hidden_dim
            self.f = nn.ModuleList([MLPBlock(in_dim, cfg.hidden_dim) for _ in range(cfg.k_steps)])
        elif cfg.module == "rnn":
            in_dim = cfg.n_comm_symbols + (cfg.hidden_dim if cfg.use_skip else 0)
            self.f = nn.RNNCell(in_dim, cfg.hidden_dim)
        elif cfg.module == "lstm":
            in_dim = cfg.n_comm_symbols + (cfg.hidden_dim if cfg.use_skip else 0)
            self.f = nn.LSTMCell(in_dim, cfg.hidden_dim)
        else:
            raise ValueError(cfg.module)

        self.comm_head = nn.Linear(cfg.hidden_dim, cfg.n_comm_symbols)
        self.pi = nn.Linear(cfg.hidden_dim, cfg.n_actions)
        self.v = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        comm_mode: str = "sample",
        state=None,
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = obs.shape
        h0 = torch.tanh(self.encoder(obs))
        h = h0

        # communication channel c is the OR of one-hot symbols from all agents
        c = torch.zeros(B, N, self.cfg.n_comm_symbols, device=obs.device)

        # For LSTM module inside the comm-steps (NOT temporal across env steps here)
        lstm_cell = None
        if self.cfg.module == "lstm":
            lstm_cell = torch.zeros(B, N, self.cfg.hidden_dim, device=obs.device)

        comm_logp_total = torch.zeros(B, N, device=obs.device)

        for i in range(self.cfg.k_steps):
            # emit symbols
            comm_logits = self.comm_head(h)  # [B,N,S]
            dist_w = torch.distributions.Categorical(logits=comm_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9))

            if comm_mode == "greedy":
                w = dist_w.probs.argmax(dim=-1)
            else:
                w = dist_w.sample()

            comm_logp_total = comm_logp_total + dist_w.log_prob(w) * mask

            w_onehot = torch.zeros(B, N, self.cfg.n_comm_symbols, device=obs.device)
            w_onehot.scatter_(-1, w.unsqueeze(-1), 1.0)

            # OR aggregation (bag-of-words)
            c_next = torch.clamp(w_onehot.sum(dim=1, keepdim=True).repeat(1, N, 1), 0.0, 1.0)

            # update hidden
            if self.cfg.module == "mlp":
                inp = torch.cat([h, c_next] + ([h0] if self.cfg.use_skip else []), dim=-1)
                h = torch.tanh(self.f[i](inp))
            elif self.cfg.module == "rnn":
                inp = torch.cat([c_next] + ([h0] if self.cfg.use_skip else []), dim=-1)
                inp2 = inp.reshape(B * N, -1)
                h2 = h.reshape(B * N, -1)
                h = self.f(inp2, h2).reshape(B, N, -1)
            else:  # lstm
                inp = torch.cat([c_next] + ([h0] if self.cfg.use_skip else []), dim=-1)
                inp2 = inp.reshape(B * N, -1)
                h2 = h.reshape(B * N, -1)
                c2 = lstm_cell.reshape(B * N, -1)
                h3, c3 = self.f(inp2, (h2, c2))
                h = h3.reshape(B, N, -1)
                lstm_cell = c3.reshape(B, N, -1)

            c = c_next

        logits = self.pi(h)
        baseline = self.v(h).squeeze(-1)

        logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        baseline = baseline * mask

        return {
            "action_logits": logits,
            "baseline": baseline,
            "comm_logp": comm_logp_total,
            "comm_vec": c,  # last OR-aggregated channel (useful for debugging)
            "next_state": None,
        }


class CommNetContinuous(nn.Module):
    """Continuous CommNet (paper main model).

    Important implementation detail vs many simplified re-implementations:
    - For the MLP module, the paper defines f^i (i=0..K), so we keep *separate*
      parameters per communication step i.
    - Communication is a broadcast mean (Eq. 2).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.obs_dim, cfg.hidden_dim)

        # In the paper, c and h have same dimension.
        self.comm_dim = cfg.hidden_dim

        if cfg.module == "mlp":
            in_dim = cfg.hidden_dim + self.comm_dim
            if cfg.use_skip:
                in_dim += cfg.hidden_dim
            self.f = nn.ModuleList([MLPBlock(in_dim, cfg.hidden_dim) for _ in range(cfg.k_steps)])
        elif cfg.module == "rnn":
            in_dim = self.comm_dim + (cfg.hidden_dim if cfg.use_skip else 0)
            self.f = nn.RNNCell(in_dim, cfg.hidden_dim)
        elif cfg.module == "lstm":
            in_dim = self.comm_dim + (cfg.hidden_dim if cfg.use_skip else 0)
            self.f = nn.LSTMCell(in_dim, cfg.hidden_dim)
        else:
            raise ValueError(cfg.module)

        self.pi = nn.Linear(cfg.hidden_dim, cfg.n_actions)
        self.v = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, state=None) -> Dict[str, torch.Tensor]:
        B, N, _ = obs.shape
        h0 = torch.tanh(self.encoder(obs))  # [B,N,H]
        h = h0

        # initial comm vector c^0 = 0 (paper)
        c = torch.zeros(B, N, self.comm_dim, device=obs.device)

        lstm_cell = None
        if self.cfg.module == "lstm":
            lstm_cell = torch.zeros(B, N, self.cfg.hidden_dim, device=obs.device)

        for i in range(self.cfg.k_steps):
            if self.cfg.module == "mlp":
                inp = torch.cat([h, c] + ([h0] if self.cfg.use_skip else []), dim=-1)
                h_next = torch.tanh(self.f[i](inp))
            elif self.cfg.module == "rnn":
                # Recurrent module over communication steps (shared weights):
                # hidden state is h, input is [c, (skip h0)].
                inp = torch.cat([c] + ([h0] if self.cfg.use_skip else []), dim=-1)
                inp2 = inp.reshape(B * N, -1)
                h2 = h.reshape(B * N, -1)
                h_next = self.f(inp2, h2).reshape(B, N, -1)
            else:  # lstm
                inp = torch.cat([c] + ([h0] if self.cfg.use_skip else []), dim=-1)
                inp2 = inp.reshape(B * N, -1)
                h2 = h.reshape(B * N, -1)
                c2 = lstm_cell.reshape(B * N, -1)
                h3, c3 = self.f(inp2, (h2, c2))
                h_next = h3.reshape(B, N, -1)
                lstm_cell = c3.reshape(B, N, -1)

            # compute comm vector for next step (Eq. 2)
            h_masked = h_next * mask.unsqueeze(-1)
            sum_h = h_masked.sum(dim=1, keepdim=True)  # [B,1,H]
            count = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1.0)  # [B,1,1]
            count_other = (count - 1.0).clamp_min(1.0)
            c = (sum_h - h_masked) / count_other

            h = h_next

        logits = self.pi(h)
        baseline = self.v(h).squeeze(-1)

        logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        baseline = baseline * mask

        # Expose comm vectors for Appendix D analysis (average norm over grid).
        return {
            "action_logits": logits,
            "baseline": baseline,
            "comm_vec": c,     # final communication vector per agent
            "hidden": h,       # final hidden per agent (sometimes useful)
            "next_state": None,
        }


def make_model(model_name: str, cfg: ModelConfig) -> nn.Module:
    if model_name == "independent":
        return IndependentPolicy(cfg)
    if model_name == "fully_connected":
        return FullyConnectedPolicy(cfg)
    if model_name == "discrete_comm":
        return DiscreteCommPolicy(cfg)
    if model_name == "commnet":
        return CommNetContinuous(cfg)
    raise ValueError(f"Unknown model_name: {model_name}")
