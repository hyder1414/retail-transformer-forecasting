from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.config.data_config import CONFIG


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensure types are clean and return (df, feature_cols).
    """
    # ---- categorical encodings ----
    # Type -> numeric
    if "Type" in df.columns:
        df["Type_id"] = df["Type"].astype("category").cat.codes.astype("int16")
        df = df.drop(columns=["Type"])
    else:
        df["Type_id"] = 0

    # Store / Dept IDs as numeric features
    if "Store" in df.columns:
        df["Store_id"] = df["Store"].astype("category").cat.codes.astype("int16")
    else:
        df["Store_id"] = 0

    if "Dept" in df.columns:
        df["Dept_id"] = df["Dept"].astype("category").cat.codes.astype("int16")
    else:
        df["Dept_id"] = 0

    # Ensure boolean holiday cols become 0/1
    for col in ["IsHoliday", "IsHoliday_feat"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Desired feature set
    desired = list(CONFIG.base_feature_cols) + [
        "Type_id",
        "Store_id",
        "Dept_id",
    ]

    # Keep only existing columns
    feature_cols = [c for c in desired if c in df.columns]

    # Fill missing with 0 before scaling
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # ---- normalize continuous features ----
    continuous_cols = [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "Size",
        "temp_anomaly",
    ]
    for col in continuous_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-6)

    # Store/Dept/Type IDs stay as small integers; that's fine
    return df, feature_cols




def make_windows_for_group(
    values: np.ndarray,
    target: np.ndarray,
    input_length: int,
    output_length: int,
):
    """
    values: (T, F) full feature matrix for a single (Store, Dept)
    target: (T,) target series for same group
    """
    T = len(target)
    min_len = input_length + output_length
    if T < min_len:
        return [], []

    X_list = []
    y_list = []

    # sliding window
    for t in range(input_length, T - output_length + 1):
        x_window = values[t - input_length : t, :]        # shape (L, F)
        y_window = target[t : t + output_length]          # shape (H,)
        X_list.append(x_window)
        y_list.append(y_window)

    return X_list, y_list


def main():
    input_length = CONFIG.input_length
    output_length = CONFIG.output_length

    print("Loading joined dataset...")
    df = pd.read_parquet(INTERIM_DIR / "joined.parquet")

    # sort by panel id + time
    df = df.sort_values(list(CONFIG.id_cols) + [CONFIG.time_col]).reset_index(drop=True)

    print("Preparing features...")
    df, feature_cols = prepare_features(df)

    X_all = []
    y_all = []

    print("Building windows per (Store, Dept)...")
    for keys, group in df.groupby(list(CONFIG.id_cols)):
        group = group.reset_index(drop=True)

        values = group[feature_cols].to_numpy(dtype=np.float32)
        target_raw = group[CONFIG.target_col].to_numpy(dtype=np.float32)
        # clip negative values to 0 before log1p
        target_clipped = np.clip(target_raw, a_min=0.0, a_max=None)
        target = np.log1p(target_clipped)


        X_list, y_list = make_windows_for_group(
            values,
            target,
            input_length=input_length,
            output_length=output_length,
        )

        if not X_list:
            continue

        X_all.extend(X_list)
        y_all.extend(y_list)

    X = np.stack(X_all)  # (N, L, F)
    y = np.stack(y_all)  # (N, H)

    print("Final window shapes:")
    print("  X:", X.shape)  # (num_windows, input_length, num_features)
    print("  y:", y.shape)  # (num_windows, output_length)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"windows_L{input_length}_H{output_length}.npz"

    print(f"Saving to {out_path}")
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        feature_cols=np.array(feature_cols),
        input_length=np.array(input_length),
        output_length=np.array(output_length),
    )

    print("Done.")


if __name__ == "__main__":
    main()
