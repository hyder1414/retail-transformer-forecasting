# src/eval/eval_gbdt_wmae.py
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def weighted_mae(y_true, y_pred, is_holiday, holiday_weight=5.0, non_holiday_weight=1.0):
    """
    Competition-style Weighted MAE:
      - weight = 5 for holiday weeks
      - weight = 1 otherwise
    """
    w = np.where(is_holiday.astype(bool), holiday_weight, non_holiday_weight)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)


def main(val_start_date: str = "2012-01-01"):
    parquet_path = INTERIM_DIR / "hahrt_with_residuals.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run train_hahrt.py first so "
            "train_gbdt_and_compute_residuals() creates it."
        )

    df = pd.read_parquet(parquet_path)

    # Same validation split as HAHRT
    val_start = pd.to_datetime(val_start_date)
    df_val = df[df["Date"] >= val_start].copy()

    # Keep only rows where baseline prediction exists
    df_val = df_val[df_val["Baseline_Pred"].notna()]

    y_true = df_val["Weekly_Sales"].to_numpy(dtype=float)
    y_pred_gbdt = df_val["Baseline_Pred"].to_numpy(dtype=float)
    is_holiday = df_val["IsHoliday"].to_numpy()

    wmae_gbdt = weighted_mae(y_true, y_pred_gbdt, is_holiday)

    print(f"Validation WMAE (GBDT baseline only): {wmae_gbdt:.2f}")


if __name__ == "__main__":
    main()
