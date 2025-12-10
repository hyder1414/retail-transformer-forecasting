# src/features/build_baseline_features.py

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import INTERIM_DIR


def main():
    joined_path = INTERIM_DIR / "joined.parquet"
    print(f"Loading joined dataset from: {joined_path}")
    df = pd.read_parquet(joined_path)

    # Ensure correct types
    df["Weekly_Sales"] = df["Weekly_Sales"].astype("float32")
    df["Store"] = df["Store"].astype(int)
    df["Dept"] = df["Dept"].astype(int)

    # Make sure Date is datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    # Sort for group-based lags/rolls
    df = df.sort_values(["Store", "Dept", "Date"])

    # Raw target for baseline model
    df["Weekly_Sales_raw"] = df["Weekly_Sales"].astype("float32")

    group_cols = ["Store", "Dept"]

    # --- Lag features on raw sales ---
    for lag in [1, 2, 7, 52]:
        df[f"lag_{lag}"] = (
            df.groupby(group_cols)["Weekly_Sales_raw"]
            .shift(lag)
        )

    # --- Rolling means (shifted by 1 to avoid leakage) ---
    for window in [4, 8, 13]:
        df[f"roll_mean_{window}"] = (
            df.groupby(group_cols)["Weekly_Sales_raw"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # --- Encode "Type" as category id for the tree model ---
    if "Type" in df.columns:
        df["Type_id"] = df["Type"].astype("category").cat.codes.astype("int16")
    else:
        df["Type_id"] = 0

    # Basic calendar/external features (assumed present from joined.parquet)
    base_feature_cols = [
        "IsHoliday",
        "CPI",
        "Fuel_Price",
        "Unemployment",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "Size",
        "Temperature",
        "dayofweek",
        "weekofyear",
        "month",
        "year",
        "temp_anomaly",
        "Type_id",
    ]

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    roll_cols = [c for c in df.columns if c.startswith("roll_mean_")]

    feature_cols = base_feature_cols + lag_cols + roll_cols

    # Keep only rows with complete feature info (drop early lag/roll NaNs)
    cols_to_keep = ["Store", "Dept", "Date", "Weekly_Sales_raw"] + feature_cols
    df_feat = df[cols_to_keep].dropna().reset_index(drop=True)

    out_path = INTERIM_DIR / "baseline_features.parquet"
    print(f"Saving baseline features to: {out_path}")
    df_feat.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
