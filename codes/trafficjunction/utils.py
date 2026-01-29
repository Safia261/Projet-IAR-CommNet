
from dataclasses import dataclass
import os, json, time
import numpy as np
import torch
import random

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class TrainStats:
    step: int
    mean_return: float
    mean_fail: float
    loss: float

def masked_mean(x, mask, dim=None, eps=1e-8):
    # mask is 0/1 broadcastable to x
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)

    return num / den

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")
