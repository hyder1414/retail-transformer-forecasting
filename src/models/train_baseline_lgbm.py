# src/models/train_baseline_lgbm.py

from math import sqrt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config.paths import INTERIM_DIR


def main():
    feat_path = INTERIM_DIR / "baseline_features.parquet"
    print(f"Loading baseline features from: {feat_path}")
    df = pd.read_parquet(feat_path)

    target_col = "Weekly_Sales_raw"
    id_cols = ["Store", "Dept", "Date"]

    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    # Sort by date and do a simple time-based train/val split
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    split_idx = int(0.8 * n)

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype("float32")

    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype("float32")

    print(
        f"Training XGBoost baseline on {len(train_df)} rows, "
        f"validating on {len(val_df)} rows..."
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",   # good default on CPU
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        verbose=50,
    )

    # Predict for ALL rows (train + val) for later residual modeling
    X_full = df[feature_cols]
    y_full = df[target_col].astype("float32")
    df["Baseline_Pred"] = model.predict(X_full)

    # Evaluate baseline quality (overall)
    rmse = sqrt(mean_squared_error(y_full, df["Baseline_Pred"]))
    mae = mean_absolute_error(y_full, df["Baseline_Pred"])

    # WMAE (5x weight on holidays)
    if "IsHoliday" in df.columns:
        weights = df["IsHoliday"].apply(lambda x: 5.0 if x else 1.0)
    else:
        weights = 1.0

    wmae = (weights * (y_full - df["Baseline_Pred"]).abs()).sum() / weights.sum()

    print("XGBoost Baseline Metrics:")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE  = {mae:.2f}")
    print(f"  WMAE = {wmae:.2f}")

    # Merge baseline preds back into the original joined.parquet
    joined_path = INTERIM_DIR / "joined.parquet"
    print(f"Loading original joined data from: {joined_path}")
    joined = pd.read_parquet(joined_path)

    merged = joined.merge(
        df[["Store", "Dept", "Date", "Baseline_Pred"]],
        on=["Store", "Dept", "Date"],
        how="left",
    )

    out_path = INTERIM_DIR / "joined_with_baseline.parquet"
    print(f"Saving joined data with baseline predictions to: {out_path}")
    merged.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
