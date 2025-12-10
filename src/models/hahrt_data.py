# src/models/hahrt_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def _encode_store_type(series: pd.Series) -> pd.Series:
    mapping = {v: i for i, v in enumerate(sorted(series.dropna().unique()))}
    return series.map(mapping).astype("int64"), mapping


def prepare_walmart_dataframe() -> pd.DataFrame:
    """
    Load and merge train.csv, features.csv, stores.csv.
    Engineer basic calendar + weather anomaly features and lags/rollings.
    """
    train_path = RAW_DIR / "train.csv"
    features_path = RAW_DIR / "features.csv"
    stores_path = RAW_DIR / "stores.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Expected train.csv at {train_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Expected features.csv at {features_path}")
    if not stores_path.exists():
        raise FileNotFoundError(f"Expected stores.csv at {stores_path}")

    train = pd.read_csv(train_path)
    features = pd.read_csv(features_path)
    stores = pd.read_csv(stores_path)

    # Parse dates
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    # Drop IsHoliday from features to avoid duplicate col on merge
    if "IsHoliday" in features.columns:
        features = features.drop(columns=["IsHoliday"])

    # Merge
    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")

    # Basic calendar features
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype("int64")
    df["month"] = df["Date"].dt.month.astype("int64")
    df["year"] = df["Date"].dt.year.astype("int64")
    df["year_index"] = df["year"] - df["year"].min()

    # Weather anomaly: Temperature - average temp per week-of-year
    if "Temperature" in df.columns:
        week_temp_mean = df.groupby("week_of_year")["Temperature"].transform("mean")
        df["Temp_Anomaly"] = df["Temperature"] - week_temp_mean
    else:
        df["Temp_Anomaly"] = 0.0

    # Encode store type
    df["StoreTypeIdx"], store_type_mapping = _encode_store_type(df["Type"])

    # Normalize store size
    if "Size" in df.columns:
        size = df["Size"].astype("float32")
        df["SizeNorm"] = (size - size.mean()) / (size.std() + 1e-6)
    else:
        df["SizeNorm"] = 0.0

    # Sort for group-based features
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    # Lags & rolling means of Weekly_Sales
    for lag in [1, 2, 7, 52]:
        df[f"lag_{lag}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
        )

    for window in [4, 8, 13]:
        df[f"roll_mean_{window}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # Integer indices for store/dept
    df["StoreIdx"] = df["Store"].astype("int64")  # stores are 1..N
    dept_ids = sorted(df["Dept"].unique())
    dept_to_idx = {d: i for i, d in enumerate(dept_ids)}
    df["DeptIdx"] = df["Dept"].map(dept_to_idx).astype("int64")

    return df


def train_gbdt_and_compute_residuals(
    df: pd.DataFrame,
    val_start_date: str = "2012-01-01",
) -> pd.DataFrame:
    """
    Train a Gradient Boosting baseline on row-level tabular features and compute residuals.
    Residuals are saved in df["Residual"], predictions in df["Baseline_Pred"].

    This version uses scikit-learn's GradientBoostingRegressor to avoid native LightGBM
    dependencies (libomp) that are painful on macOS.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    feature_cols = [
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

    # Some columns might be missing (e.g. MarkDowns); fill with 0 so we don't lose rows
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    used_cols = feature_cols + ["Weekly_Sales", "Date"]
    df_model = df[used_cols].copy()

    # Replace any remaining NaNs in features with 0
    df_model[feature_cols] = df_model[feature_cols].astype("float32").fillna(0.0)

    X = df_model[feature_cols].values.astype("float32")
    y = df_model["Weekly_Sales"].astype("float32").values
    date_values = df_model["Date"].values

    val_start = np.datetime64(val_start_date)
    train_mask = date_values < val_start
    # If val_start is outside range, use all for training
    if train_mask.sum() == 0:
        train_mask[:] = True

    X_train = X[train_mask]
    y_train = y[train_mask]

    # Gradient Boosting baseline (tree-based, like LightGBM but pure scikit-learn)
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        max_features=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict on all df_model rows (train + val)
    y_pred_all = model.predict(X)
    residuals = y - y_pred_all

    # Initialize columns if they don't exist
    if "Baseline_Pred" not in df.columns:
        df["Baseline_Pred"] = np.nan
    if "Residual" not in df.columns:
        df["Residual"] = np.nan

    # Map predictions/residuals back to original df indices
    df.loc[df_model.index, "Baseline_Pred"] = y_pred_all
    df.loc[df_model.index, "Residual"] = residuals

    # Persist to interim for debugging / reuse
    out_path = INTERIM_DIR / "hahrt_with_residuals.parquet"
    df.to_parquet(out_path, index=False)

    return df


@dataclass
class SequenceSamples:
    x_cont: np.ndarray
    week_of_year: np.ndarray
    year_index: np.ndarray
    is_holiday: np.ndarray
    store_idx: np.ndarray
    dept_idx: np.ndarray
    store_type_idx: np.ndarray
    size_norm: np.ndarray
    target_resid: np.ndarray
    target_weight: np.ndarray
    baseline_pred_target: np.ndarray
    target_y: np.ndarray


class HAHRTSequenceDataset(Dataset):
    def __init__(self, samples: SequenceSamples):
        self.samples = samples
        self.num_samples = samples.x_cont.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        s = self.samples
        return {
            "x_cont": torch.from_numpy(s.x_cont[idx]),  # [L, D]
            "week_of_year": torch.from_numpy(s.week_of_year[idx]),  # [L]
            "year_idx": torch.from_numpy(s.year_index[idx]),  # [L]
            "is_holiday": torch.from_numpy(s.is_holiday[idx]),  # [L]
            "store_idx": torch.tensor(s.store_idx[idx], dtype=torch.long),
            "dept_idx": torch.tensor(s.dept_idx[idx], dtype=torch.long),
            "store_type_idx": torch.tensor(s.store_type_idx[idx], dtype=torch.long),
            "size_norm": torch.tensor(s.size_norm[idx], dtype=torch.float32),
            "target_resid": torch.tensor(s.target_resid[idx], dtype=torch.float32),
            "target_weight": torch.tensor(
                s.target_weight[idx], dtype=torch.float32
            ),
            "baseline_pred_target": torch.tensor(
                s.baseline_pred_target[idx], dtype=torch.float32
            ),
            "target_y": torch.tensor(s.target_y[idx], dtype=torch.float32),
        }


def _build_samples_for_split(
    df: pd.DataFrame,
    indices: np.ndarray,
    input_window: int,
    cont_feature_cols: List[str],
) -> SequenceSamples:
    """
    Build sliding-window sequence samples for the given row indices.
    Indices refer to rows in df; they must be sorted by (Store, Dept, Date).
    """
    sequences_x: List[np.ndarray] = []
    sequences_week: List[np.ndarray] = []
    sequences_year: List[np.ndarray] = []
    sequences_holiday: List[np.ndarray] = []
    sequences_store: List[int] = []
    sequences_dept: List[int] = []
    sequences_store_type: List[int] = []
    sequences_size: List[float] = []
    targets_resid: List[float] = []
    targets_weight: List[float] = []
    baseline_targets: List[float] = []
    target_y: List[float] = []

    # We will iterate group-wise to enforce sequence continuity
    grouped = df.loc[indices].groupby(["Store", "Dept"])

    for (_, _), g in grouped:
        g = g.sort_values("Date")
        if len(g) <= input_window:
            continue

        cont_vals = g[cont_feature_cols].astype("float32").values
        week_vals = g["week_of_year"].astype("int64").values
        year_vals = g["year_index"].astype("int64").values
        holiday_vals = g["IsHoliday"].astype("int64").values
        store_idx_vals = g["StoreIdx"].astype("int64").values
        dept_idx_vals = g["DeptIdx"].astype("int64").values
        store_type_idx_vals = g["StoreTypeIdx"].astype("int64").values
        size_vals = g["SizeNorm"].astype("float32").values
        resid_vals = g["Residual"].astype("float32").values
        weight_vals = np.where(g["IsHoliday"].values, 5.0, 1.0).astype("float32")
        baseline_vals = g["Baseline_Pred"].astype("float32").values
        target_y_vals = g["Weekly_Sales"].astype("float32").values

        for i in range(input_window, len(g)):
            # sequence covers [i-input_window, i)
            seq_slice = slice(i - input_window, i)

            # require that target has valid residual & baseline
            if np.isnan(resid_vals[i]) or np.isnan(baseline_vals[i]):
                continue

            sequences_x.append(cont_vals[seq_slice])
            sequences_week.append(week_vals[seq_slice])
            sequences_year.append(year_vals[seq_slice])
            sequences_holiday.append(holiday_vals[seq_slice])
            sequences_store.append(int(store_idx_vals[i]))
            sequences_dept.append(int(dept_idx_vals[i]))
            sequences_store_type.append(int(store_type_idx_vals[i]))
            sequences_size.append(float(size_vals[i]))
            targets_resid.append(float(resid_vals[i]))
            targets_weight.append(float(weight_vals[i]))
            baseline_targets.append(float(baseline_vals[i]))
            target_y.append(float(target_y_vals[i]))

    if not sequences_x:
        raise RuntimeError("No sequences built; check data and input_window size.")

    x_cont = np.stack(sequences_x, axis=0)
    week = np.stack(sequences_week, axis=0)
    year = np.stack(sequences_year, axis=0)
    holiday = np.stack(sequences_holiday, axis=0)
    store_idx = np.array(sequences_store, dtype="int64")
    dept_idx = np.array(sequences_dept, dtype="int64")
    store_type_idx = np.array(sequences_store_type, dtype="int64")
    size_norm = np.array(sequences_size, dtype="float32")
    target_resid = np.array(targets_resid, dtype="float32")
    target_weight = np.array(targets_weight, dtype="float32")
    baseline_pred_target = np.array(baseline_targets, dtype="float32")
    target_y_arr = np.array(target_y, dtype="float32")

    return SequenceSamples(
        x_cont=x_cont,
        week_of_year=week,
        year_index=year,
        is_holiday=holiday,
        store_idx=store_idx,
        dept_idx=dept_idx,
        store_type_idx=store_type_idx,
        size_norm=size_norm,
        target_resid=target_resid,
        target_weight=target_weight,
        baseline_pred_target=baseline_pred_target,
        target_y=target_y_arr,
    )


def build_sequence_datasets(
    df: pd.DataFrame,
    input_window: int = 12,
    val_start_date: str = "2012-01-01",
) -> Tuple[HAHRTSequenceDataset, HAHRTSequenceDataset, Dict]:
    """
    Build train/val HAHRTSequenceDataset objects from df that already has residuals.
    """
    # Ensure residuals are computed
    if "Residual" not in df.columns or df["Residual"].isna().all():
        raise ValueError(
            "DataFrame must have non-null 'Residual' column. "
            "Run train_gbdt_and_compute_residuals first."
        )

    # Continuous covariates fed into Transformer
    cont_feature_cols = [
        "Residual",  # most recent residuals in the window
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Temp_Anomaly",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "lag_1",
        "lag_2",
        "lag_7",
        "lag_52",
        "roll_mean_4",
        "roll_mean_8",
        "roll_mean_13",
    ]

    # Ensure all cont feature columns exist
    for col in cont_feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # IMPORTANT: fill NaNs in continuous features instead of dropping rows
    df[cont_feature_cols] = df[cont_feature_cols].fillna(0.0)

    # Work only on rows where residual & baseline are available
    df_valid = df.dropna(subset=["Residual", "Baseline_Pred"]).copy()

    # Sort and keep original index in a column named "index"
    df_valid = df_valid.sort_values(["Store", "Dept", "Date"]).reset_index()

    all_indices = df_valid["index"].values   # original df indices
    dates = df_valid["Date"].values
    val_start = np.datetime64(val_start_date)

    train_mask = dates < val_start
    if train_mask.sum() == 0:
        train_mask[:] = True

    train_indices = all_indices[train_mask]
    val_indices = all_indices[~train_mask]

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError(
            f"Train/val split empty: "
            f"{len(train_indices)} train rows, {len(val_indices)} val rows. "
            f"Check val_start_date={val_start_date}."
        )

    train_samples = _build_samples_for_split(
        df, train_indices, input_window, cont_feature_cols
    )
    val_samples = _build_samples_for_split(
        df, val_indices, input_window, cont_feature_cols
    )

    train_ds = HAHRTSequenceDataset(train_samples)
    val_ds = HAHRTSequenceDataset(val_samples)

    meta = {
        "num_stores": int(df["StoreIdx"].max()) + 1,
        "num_depts": int(df["DeptIdx"].max()) + 1,
        "num_store_types": int(df["StoreTypeIdx"].max()) + 1,
        "max_week_of_year": int(df["week_of_year"].max()),
        "max_year_index": int(df["year_index"].max()),
        "input_dim": len(cont_feature_cols),
    }

    return train_ds, val_ds, meta
