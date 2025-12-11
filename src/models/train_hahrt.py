# src/models/train_hahrt.py
import argparse
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .hahrt_data import (
    prepare_walmart_dataframe,
    train_gbdt_and_compute_residuals,
    build_sequence_datasets,
    DATA_DIR,
)
from .hahrt_model import HAHRTModel, HAHRTConfig


def weighted_mae_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Weighted MAE aligned with the Walmart Kaggle WMAE metric:
      sum(w_i * |y_i - yhat_i|) / sum(w_i),
    with higher weights for holiday weeks.
    """
    eps = 1e-8
    abs_err = torch.abs(pred - target)
    num = torch.sum(weight * abs_err)
    denom = torch.sum(weight) + eps
    return num / denom


def compute_wmae_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> float:
    with torch.no_grad():
        loss = weighted_mae_loss(pred, target, weight)
    return float(loss.cpu().item())


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_mse: float = 0.0,  # kept for CLI compatibility, unused now
) -> float:
    """
    Train for one epoch using PURE weighted MAE (WMAE) on residuals.
    Returns the average train WMAE across batches.
    """
    model.train()
    total_wmae = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        batch = move_batch_to_device(batch, device)

        # HAHRTModel returns residual prediction [B]
        pred_resid = model(batch)  # [B]

        target_resid = batch["target_resid"]     # [B]
        target_weight = batch["target_weight"]   # [B]

        loss = weighted_mae_loss(pred_resid, target_resid, target_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_wmae += float(loss.detach().cpu().item())
        num_batches += 1

    return total_wmae / max(1, num_batches)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate on validation set:
      - WMAE on residuals
      - WMAE on final prediction (baseline + residual)
    """
    model.eval()
    all_wmae_resid = []
    all_wmae_final = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            pred_resid = model(batch)  # [B]

            target_resid = batch["target_resid"]
            target_weight = batch["target_weight"]
            baseline_pred = batch["baseline_pred_target"]
            target_y = batch["target_y"]

            # Residual-level WMAE
            wmae_resid = compute_wmae_metric(
                pred_resid, target_resid, target_weight
            )
            all_wmae_resid.append(wmae_resid)

            # Final sales prediction: baseline + residual
            final_pred = baseline_pred + pred_resid
            wmae_final = compute_wmae_metric(
                final_pred, target_y, target_weight
            )
            all_wmae_final.append(wmae_final)

    return {
        "wmae_resid": float(np.mean(all_wmae_resid)),
        "wmae_final": float(np.mean(all_wmae_final)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train HAHRT (Holiday-Aware Hierarchical Residual Transformer)"
    )
    parser.add_argument(
        "--val_start_date",
        type=str,
        default="2012-01-01",
        help="Validation start date (inclusive)",
    )
    parser.add_argument(
        "--input_window",
        type=int,
        default=24,
        help="Number of past weeks in each input sequence",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=0.0,
        help="(Unused) kept for backward compatibility. Training uses pure WMAE.",
    )
    # Model hyperparameters
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--holiday_bias_strength",
        type=float,
        default=1.5,
        help="Attention logit bias toward holiday timesteps",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading and preparing Walmart data...")
    df = prepare_walmart_dataframe()

    # Strong GBDT baseline: this is the 'real' production baseline
    df = train_gbdt_and_compute_residuals(df, val_start_date=args.val_start_date)

    print("Building HAHRT sequence datasets...")
    train_ds, val_ds, meta = build_sequence_datasets(
        df, input_window=args.input_window, val_start_date=args.val_start_date
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Instantiating HAHRT model...")
    cfg = HAHRTConfig(
        input_dim=meta["input_dim"],
        num_stores=meta["num_stores"],
        num_depts=meta["num_depts"],
        num_store_types=meta["num_store_types"],
        max_week_of_year=meta["max_week_of_year"],
        max_year_index=meta["max_year_index"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        holiday_bias_strength=args.holiday_bias_strength,
    )
    model = HAHRTModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_wmae = float("inf")
    best_model_path = DATA_DIR / "interim" / "hahrt_best_model.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_wmae = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_mse=args.lambda_mse,
        )
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train WMAE (resid): {train_wmae:.2f} | "
            f"Val WMAE (resid): {metrics['wmae_resid']:.2f} | "
            f"Val WMAE (final): {metrics['wmae_final']:.2f}"
        )

        if metrics["wmae_final"] < best_val_wmae:
            best_val_wmae = metrics["wmae_final"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")

    print("Training complete.")
    print(f"Best validation WMAE (final sales): {best_val_wmae:.2f}")


if __name__ == "__main__":
    main()
