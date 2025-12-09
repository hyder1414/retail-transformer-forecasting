from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.features.dataset import WindowDataset
from src.config.data_config import CONFIG


def naive_last_value_baseline_test_only(batch_size: int = 256):
    # Use the same underlying .npz and the same random split as training
    test_ds = WindowDataset(split="test")
    feature_cols = test_ds.feature_cols
    target_idx = feature_cols.index("Weekly_Sales")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    for xb, yb in test_loader:
        # xb: (B, L, F), yb: (B, H)
        last_target = xb[:, -1, target_idx]          # (B,)
        y_pred = last_target.unsqueeze(1).repeat(1, yb.size(1))  # (B, H)

        all_preds.append(y_pred)
        all_targets.append(yb)

    preds_full = torch.cat(all_preds, dim=0)
    targets_full = torch.cat(all_targets, dim=0)

    mse = torch.mean((preds_full - targets_full) ** 2).item()
    rmse = torch.sqrt(torch.mean((preds_full - targets_full) ** 2)).item()
    mae = torch.mean(torch.abs(preds_full - targets_full)).item()

    print("Naive last-value baseline on TEST split:")
    print(f"  MSE = {mse:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")


if __name__ == "__main__":
    naive_last_value_baseline_test_only()
