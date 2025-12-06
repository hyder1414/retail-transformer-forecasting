# src/utils/metrics.py
import os
import json
import numpy as np
import torch

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles):
    """
    pred: [B, H, Q]
    target: [B, H]
    """
    B, H, Q = pred.shape
    t = target.unsqueeze(-1).expand(-1, -1, Q)
    qs = torch.tensor(quantiles, device=pred.device).view(1,1,Q)
    diff = t - pred
    loss = torch.maximum(qs*diff, (qs-1)*diff)
    return loss.mean()

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def save_metrics(path: str, d: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
