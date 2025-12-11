# src/models/train_hahrt_rc_gbdt.py

import argparse
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor

from .hahrt_data import (
    prepare_walmart_dataframe,
    build_sequence_datasets,
    DATA_DIR,
    INTERIM_DIR,
)
from .hahrt_model import HAHRTModel, HAHRTConfig


def weighted_mae_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Weighted MAE:
        sum_i w_i * |y_i - yhat_i| / sum_i w_i
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
) -> float:
    """
    Pure WMAE training on residuals for diagnostic run.
    """
    model.train()
    total_wmae = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()
        batch = move_batch_to_device(batch, device)

        pred_resid = model(batch)  # [B]
        target_resid = batch["target_resid"]
        target_weight = batch["target_weight"]

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
    Evaluate residual WMAE and final WMAE (baseline + residual).
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

            wmae_resid = compute_wmae_metric(
                pred_resid, target_resid, target_weight
            )
            all_wmae_resid.append(wmae_resid)

            final_pred = baseline_pred + pred_resid
            wmae_final = compute_wmae_metric(
                final_pred, target_y, target_weight
            )
            all_wmae_final.append(wmae_final)

    return {
        "wmae_resid": float(np.mean(all_wmae_resid)),
        "wmae_final": float(np.mean(all_wmae_final)),
    }


def train_reduced_capacity_gbdt_and_compute_residuals(
    df, val_start_date: str = "2012-01-01"
):
    """
    Diagnostic: train a reduced-capacity GBDT (RC-GBDT) baseline
    to visualize more gradual learning in the residual Transformer.

    This uses fewer trees and shallower depth than the main baseline.
    """
    feature_cols: List[str] = [
        # identifiers
        "StoreIdx",
        "DeptIdx",
        "StoreTypeIdx",
        "SizeNorm",
        # calendar
        "week_of_year",
        "month",
        "year_index",
        "IsHoliday",
        # dynamics
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Temp_Anomaly",
        # markdowns
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        # lags
        "lag_1",
        "lag_2",
        "lag_7",
        "lag_52",
        "roll_mean_4",
        "roll_mean_8",
        "roll_mean_13",
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    used_cols = feature_cols + ["Weekly_Sales", "Date"]
    df_model = df[used_cols].copy()
    df_model[feature_cols] = df_model[feature_cols].astype("float32").fillna(0.0)

    X = df_model[feature_cols].values.astype("float32")
    y = df_model["Weekly_Sales"].astype("float32").values
    date_values = df_model["Date"].values

    val_start = np.datetime64(val_start_date)
    train_mask = date_values < val_start
    if train_mask.sum() == 0:
        train_mask[:] = True

    X_train = X[train_mask]
    y_train = y[train_mask]

    # Reduced-capacity GBDT (RC-GBDT) for diagnostic ablation
    model = GradientBoostingRegressor(
        n_estimators=30,     # fewer trees
        learning_rate=0.15, # slightly higher LR -> rougher fit
        max_depth=1,        # very shallow trees (stumps)
        subsample=0.8,
        max_features=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Diagnostic: compute RC-GBDT-only WMAE on validation split ---
    val_mask = date_values >= val_start
    if val_mask.sum() > 0:
        y_val = y[val_mask]
        y_val_pred = model.predict(X[val_mask])

        is_holiday_val = df_model["IsHoliday"].values[val_mask].astype(bool)
        w_val = np.where(is_holiday_val, 5.0, 1.0).astype("float32")

        abs_err = np.abs(y_val - y_val_pred)
        wmae_val = float((w_val * abs_err).sum() / w_val.sum())
        print(f"[RC-GBDT] Baseline-only Val WMAE: {wmae_val:.2f}")
    else:
        print("[RC-GBDT] Warning: no validation rows found for WMAE calc")

    # Compute residuals for all rows
    y_pred_all = model.predict(X)
    residuals = y - y_pred_all

    if "Baseline_Pred" not in df.columns:
        df["Baseline_Pred"] = np.nan
    if "Residual" not in df.columns:
        df["Residual"] = np.nan

    df.loc[df_model.index, "Baseline_Pred"] = y_pred_all
    df.loc[df_model.index, "Residual"] = residuals

    out_path = INTERIM_DIR / "hahrt_with_residuals_rc_gbdt.parquet"
    df.to_parquet(out_path, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic: Train HAHRT with a reduced-capacity GBDT baseline "
            "to visualize learning dynamics"
        )
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
        default=15,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
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
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading and preparing Walmart data...")
    df = prepare_walmart_dataframe()

    print("Training reduced-capacity GBDT (RC-GBDT) baseline (diagnostic)...")
    df = train_reduced_capacity_gbdt_and_compute_residuals(
        df, val_start_date=args.val_start_date
    )

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

    print("Instantiating HAHRT model (diagnostic)...")
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
    best_model_path = DATA_DIR / "interim" / "hahrt_best_model_rc_gbdt.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_wmae = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)

        print(
            f"[RC-GBDT] Epoch {epoch:02d} | "
            f"Train WMAE (resid): {train_wmae:.2f} | "
            f"Val WMAE (resid): {metrics['wmae_resid']:.2f} | "
            f"Val WMAE (final): {metrics['wmae_final']:.2f}"
        )

        if metrics["wmae_final"] < best_val_wmae:
            best_val_wmae = metrics["wmae_final"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best diagnostic model saved to {best_model_path}")

    print("Diagnostic training complete.")
    print(f"Best validation WMAE (final, RC-GBDT): {best_val_wmae:.2f}")


if __name__ == "__main__":
    main()
