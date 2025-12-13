# src/models/train_hahrt.py
import argparse
import json
from pathlib import Path
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def weighted_mse_log_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    MSE in log1p space with holiday weighting.
    This stabilizes very large sales values and complements WMAE.
    """
    eps = 1e-8
    pred_c = torch.clamp(pred, min=0.0)
    target_c = torch.clamp(target, min=0.0)
    per_sample = (torch.log1p(pred_c) - torch.log1p(target_c)) ** 2
    num = torch.sum(per_sample * weight)
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
    final_wmae_weight: float = 0.7,
    lambda_log: float = 0.05,
) -> Dict[str, float]:
    """
    Train for one epoch optimizing a blended objective:
      - holiday-weighted MAE on residuals
      - holiday-weighted MAE on final prediction (baseline + residual)
      - log-space MSE regularizer to stabilize extreme sales
    Returns average train metrics.
    """
    model.train()
    total_wmae_resid = 0.0
    total_wmae_final = 0.0
    total_mse_log = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        batch = move_batch_to_device(batch, device)

        # HAHRTModel returns residual prediction [B]
        pred_resid = model(batch)  # [B]

        target_resid = batch["target_resid"]     # [B]
        target_weight = batch["target_weight"]   # [B]
        baseline_pred = batch["baseline_pred_target"]  # [B]
        target_y = batch["target_y"]             # [B]

        final_pred = baseline_pred + pred_resid

        wmae_resid = weighted_mae_loss(pred_resid, target_resid, target_weight)
        wmae_final = weighted_mae_loss(final_pred, target_y, target_weight)
        mse_log_final = weighted_mse_log_loss(final_pred, target_y, target_weight)

        loss = (
            final_wmae_weight * wmae_final
            + (1.0 - final_wmae_weight) * wmae_resid
            + lambda_log * mse_log_final
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_wmae_resid += float(wmae_resid.detach().cpu().item())
        total_wmae_final += float(wmae_final.detach().cpu().item())
        total_mse_log += float(mse_log_final.detach().cpu().item())
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "wmae_resid": total_wmae_resid / denom,
        "wmae_final": total_wmae_final / denom,
        "mse_log": total_mse_log / denom,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate on validation set:
      - WMAE on residuals (diagnostic)
      - WMAE on final prediction (competition metric)
      - MAE / RMSE / MSE_log for reporting
    """
    model.eval()
    all_wmae_resid = []
    all_wmae_final = []
    all_mae_final = []
    all_rmse_final = []
    all_mse_log_final = []

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

            mae_final = torch.mean(torch.abs(final_pred - target_y)).item()
            rmse_final = torch.sqrt(torch.mean((final_pred - target_y) ** 2)).item()
            mse_log_final = torch.mean(
                (torch.log1p(torch.clamp(final_pred, min=0.0)) - torch.log1p(torch.clamp(target_y, min=0.0))) ** 2
            ).item()

            all_mae_final.append(mae_final)
            all_rmse_final.append(rmse_final)
            all_mse_log_final.append(mse_log_final)

    return {
        "wmae_resid": float(np.mean(all_wmae_resid)),
        "wmae_final": float(np.mean(all_wmae_final)),
        "mae_final": float(np.mean(all_mae_final)),
        "rmse_final": float(np.mean(all_rmse_final)),
        "mse_log_final": float(np.mean(all_mse_log_final)),
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
        "--final_wmae_weight",
        type=float,
        default=0.7,
        help="Weight for final WMAE vs residual WMAE in the blended loss (1.0 => final only).",
    )
    parser.add_argument(
        "--lambda_log",
        type=float,
        default=0.05,
        help="Weight on the log-space MSE regularizer to stabilize very large sales.",
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
    parser.add_argument(
        "--no_film",
        action="store_true",
        help="Disable hierarchical FiLM adapters (useful for ablations / baseline transformer).",
    )
    parser.add_argument(
        "--no_local_conv",
        action="store_true",
        help="Disable the short-range convolutional stem.",
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
        use_film=not args.no_film,
        use_local_conv=not args.no_local_conv,
    )
    model = HAHRTModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_wmae = float("inf")
    best_epoch = -1
    best_val_metrics: Dict[str, float] = {}
    best_model_path = DATA_DIR / "interim" / "hahrt_best_model.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            final_wmae_weight=args.final_wmae_weight,
            lambda_log=args.lambda_log,
        )
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train WMAE (final/resid): {train_metrics['wmae_final']:.2f} / {train_metrics['wmae_resid']:.2f} | "
            f"Train MSE_log: {train_metrics['mse_log']:.4f} | "
            f"Val WMAE (final/resid): {val_metrics['wmae_final']:.2f} / {val_metrics['wmae_resid']:.2f} | "
            f"Val MAE: {val_metrics['mae_final']:.2f} | "
            f"Val MSE_log: {val_metrics['mse_log_final']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_metrics["wmae_final"] < best_val_wmae:
            best_val_wmae = val_metrics["wmae_final"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")

    # Persist metrics + history for the notebook/TA story
    best_metrics_path = REPORTS_DIR / "hahrt_best_metrics.json"
    history_path = REPORTS_DIR / "hahrt_train_history.json"
    payload = {
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "config": vars(args),
    }
    with open(best_metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")
    print(f"Best validation WMAE (final sales): {best_val_wmae:.2f}")


if __name__ == "__main__":
    main()
