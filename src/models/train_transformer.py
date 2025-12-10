from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam

from src.features.dataset import create_dataloaders
from src.models.transformer_forecaster import TimeSeriesTransformer, TransformerConfig


@dataclass
class TrainConfig:
    batch_size: int = 256   # you can try 512 if it’s not too slow
    num_workers: int = 0
    lr: float = 3e-4        # smaller LR for deeper model
    weight_decay: float = 1e-4
    max_epochs: int = 15    # give it time to learn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (M1/M2/M3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    """
    Train for one epoch using holiday-weighted MSE in log1p space.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    total_batches = len(train_loader)

    for batch_idx, (xb, yb, hb) in enumerate(train_loader):
        xb = xb.to(device)  # (B, L, F)
        yb = yb.to(device)  # (B, H)   log1p(target)
        hb = hb.to(device)  # (B,)     0.0 or 1.0 holiday flag

        optimizer.zero_grad()
        preds = model(xb)   # (B, H) in log1p space

        # per-output MSE in log space: shape (B, H)
        loss_per_h = criterion(preds, yb)          # (B, H)

        # average across horizon → per-sample loss: shape (B,)
        loss_per_sample = loss_per_h.mean(dim=1)   # (B,)

        # holiday weighting: 5x for holiday, 1x otherwise
        weights = 1.0 + 4.0 * hb                   # (B,)

        # weighted average over batch (scalar)
        loss = (weights * loss_per_sample).mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        # progress print every 100 batches
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
            avg_so_far = running_loss / n_batches
            print(
                f"Epoch {epoch} "
                f"[{batch_idx + 1}/{total_batches}] "
                f"loss={avg_so_far:.4f}"
            )

    avg_loss = running_loss / max(n_batches, 1)
    print(f"Epoch {epoch} DONE: train_loss = {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
):
    """
    Evaluate model:
      - avg MSE in log space (MSE_log)
      - RMSE, MAE, WMAE in original sales dollars
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []
    all_holidays = []

    for xb, yb, hb in loader:
        xb = xb.to(device)
        yb = yb.to(device)  # log1p(target)
        hb = hb.to(device)  # 0.0 or 1.0

        preds = model(xb)   # (B, H) in log1p space

        # criterion gives (B, H) because reduction="none"
        loss_per_h = criterion(preds, yb)       # (B, H)
        loss_batch = loss_per_h.mean()          # scalar
        running_loss += loss_batch.item()
        n_batches += 1

        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())
        all_holidays.append(hb.cpu())

    avg_mse_log = running_loss / max(n_batches, 1)

    preds_full = torch.cat(all_preds, dim=0)        # (N, H)
    targets_full = torch.cat(all_targets, dim=0)    # (N, H)
    holidays_full = torch.cat(all_holidays, dim=0)  # (N,)

    # back to original sales (undo log1p)
    preds_orig = torch.expm1(preds_full)
    targets_orig = torch.expm1(targets_full)

    # MAE / RMSE in dollars
    mae = torch.mean(torch.abs(preds_orig - targets_orig)).item()
    rmse = torch.sqrt(torch.mean((preds_orig - targets_orig) ** 2)).item()

    # WMAE: 5x for holiday, 1x otherwise
    weights = torch.where(
        holidays_full > 0.5,
        torch.tensor(5.0),
        torch.tensor(1.0),
    )  # (N,)

    # mean absolute error per sample across horizon
    ae_per_sample = torch.mean(
        torch.abs(preds_orig - targets_orig),
        dim=1,  # average over H
    )  # (N,)

    wmae = (weights * ae_per_sample).sum().item() / weights.sum().item()

    print(
        f"{split_name}_metrics: "
        f"MSE_log={avg_mse_log:.4f}, RMSE={rmse:.2f}, "
        f"MAE={mae:.2f}, WMAE={wmae:.2f}"
    )

    return avg_mse_log, rmse, mae, wmae


def main():
    train_cfg = TrainConfig()

    device = get_device()
    print("Using device:", device)

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    # infer feature_dim from dataset
    feature_dim = len(train_loader.dataset.feature_cols)
    print("Feature dim:", feature_dim)

    model_cfg = TransformerConfig(
        input_length=52,
        output_length=4,
        feature_dim=feature_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.2,
    )

    model = TimeSeriesTransformer(model_cfg).to(device)

    # IMPORTANT: reduction="none" to support weighting
    criterion = nn.MSELoss(reduction="none")

    optimizer = Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    print("Starting training...")
    for epoch in range(1, train_cfg.max_epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )
        evaluate(
            model,
            val_loader,
            criterion,
            device,
            split_name="val",
        )

    print("Final evaluation on test set:")
    evaluate(
        model,
        test_loader,
        criterion,
        device,
        split_name="test",
    )


if __name__ == "__main__":
    main()
