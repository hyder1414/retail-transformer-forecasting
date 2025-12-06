import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# -----
# Helpers to load and prepare M5 data into a single long dataframe (leakage-safe)
# and to create sliding windows for context+horizon forecasting.
# -----

def load_m5(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(os.path.join(raw_dir, "sales_train_validation.csv"))
    calendar = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))
    prices = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))
    return sales, calendar, prices


def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    val_cols = [c for c in sales.columns if c.startswith("d_")]
    df = sales[id_cols + val_cols].melt(id_vars=id_cols, var_name="d", value_name="sales")
    return df


def build_master(
    raw_dir: str,
    max_series: int = 5000,
    min_history: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Returns:
      df (long, daily): columns =
        ['id','date','sales','item_id','dept_id','cat_id','store_id','state_id',
         'dow','month','snap','price_ratio']
      static_cardinalities: dict for embedding sizes
    """
    print("  - Loading sales/calendar/prices...")
    sales, cal, prices = load_m5(raw_dir)

    # Keep only columns we need from calendar
    cal_use = cal[
        ["d", "date", "wm_yr_wk", "wday", "month", "snap_CA", "snap_TX", "snap_WI"]
    ].copy()
    cal_use["date"] = pd.to_datetime(cal_use["date"])

    # Long sales, per-day merge
    long_sales = melt_sales(sales)
    df = long_sales.merge(cal_use, on="d", how="left")

    # Merge prices at (store_id, item_id, week)
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df.rename(columns={"sell_price": "price"}, inplace=True)

    # Filter out very short series (based on non-NaN sales days)
    counts = df.groupby("id")["sales"].apply(lambda s: s.notna().sum())
    keep_ids = counts[counts >= min_history].index
    df = df[df["id"].isin(keep_ids)]

    # Subsample series for speed if requested
    if max_series and len(keep_ids) > max_series:
        keep_ids = pd.Index(np.random.default_rng(42).choice(keep_ids, size=max_series, replace=False))
        df = df[df["id"].isin(keep_ids)]

    # Preserve string state_id for SNAP logic, then encode later
    df = df.sort_values(["id", "d"]).reset_index(drop=True)
    df["sales"] = df["sales"].fillna(0.0).astype(np.float32)
    df["price"] = df["price"].astype(float)
    df["state_id_str"] = df["state_id"].astype(str)

    # Calendar derived features
    # wday in original M5 is 1..7; we keep as int 1..7
    df["dow"] = df["wday"].astype(int)
    df["month"] = df["month"].astype(int)

    # SNAP per state, using original string labels to avoid mis-mapping after encoding
    # (CA/TX/WI exist; if anything else appears, default 0)
    snap_map = {
        "CA": "snap_CA",
        "TX": "snap_TX",
        "WI": "snap_WI",
    }
    def snap_value(row):
        st = row["state_id_str"]
        col = snap_map.get(st, None)
        if col is None: 
            return 0
        val = row.get(col, 0)
        try:
            return int(val)
        except Exception:
            return 0

    df["snap"] = df.apply(snap_value, axis=1).astype(np.int32)

    # Price ratio vs past 28-day moving average (leakage-safe via shift(1))
    df["price_ma28"] = (
        df.groupby("id")["price"]
          .apply(lambda s: s.rolling(28, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    df["price_ma28"] = df.groupby("id")["price_ma28"].shift(1)
    df["price_ratio"] = (df["price"] / df["price_ma28"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["price_ratio"] = df["price_ratio"].astype(np.float32)

    # Static categorical encodings (AFTER SNAP logic)
    static_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    enc = OrdinalEncoder()
    df[static_cols] = enc.fit_transform(df[static_cols].astype(str))
    df[static_cols] = df[static_cols].astype(np.int64)
    static_cardinalities = {c: int(df[c].max() + 1) for c in static_cols}

    keep = [
        "id", "date", "sales",
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "dow", "month", "snap", "price_ratio"
    ]
    df = df[keep].sort_values(["id", "date"]).reset_index(drop=True)

    print(f"  - Keeping up to {max_series} series; merges & rolling stats complete.")
    return df, static_cardinalities


def make_windows(
    df: pd.DataFrame,
    context_len: int,
    horizon: int,
    val_last_days: int
) -> Tuple[dict, dict]:
    """
    Build sliding windows.
    Train windows come from the 'earlier' part of each series;
    Val windows come from the last `val_last_days` days (rolling-origin with horizon).
    """
    print("  - Building sliding windows for train/val...")
    Xc_train, Y_train, S_train, ids_train = [], [], [], []
    Xc_val,   Y_val,   S_val,   ids_val   = [], [], [], []

    for sid, g in df.groupby("id", sort=False):
        y = g["sales"].to_numpy(dtype=np.float32)
        X_dyn = g[["dow", "month", "snap", "price_ratio"]].to_numpy(dtype=np.float32)
        X_static = g[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].iloc[0].to_numpy(dtype=np.int64)

        T = len(y)
        # last index from which we start producing validation windows
        # so that we only use the tail "val_last_days" for val origins
        last_idx_for_val = max(0, T - val_last_days - horizon)

        for t_end in range(context_len, T - horizon + 1):
            x_start = t_end - context_len
            x_end = t_end
            y_end = t_end + horizon

            # history features: include lagged target as first channel
            x_hist = np.column_stack([
                y[x_start:x_end],          # [L]
                X_dyn[x_start:x_end, :]    # [L, 4]
            ])                              # => [L, 1+4]

            y_fut = y[t_end:y_end]          # [H]

            if t_end < last_idx_for_val:
                Xc_train.append(x_hist)
                Y_train.append(y_fut)
                S_train.append(X_static)
                ids_train.append(sid)
            else:
                Xc_val.append(x_hist)
                Y_val.append(y_fut)
                S_val.append(X_static)
                ids_val.append(sid)

    def pack(arr, shape):
        if len(arr) == 0:
            return np.zeros(shape, dtype=np.float32 if len(shape) and shape[-1] != 5 else np.int64)
        return np.stack(arr, axis=0)

    train = {
        "Xc":     pack(Xc_train, (0, context_len, 5)),
        "Y":      pack(Y_train,  (0, horizon)),
        "static": np.stack(S_train) if len(S_train) else np.zeros((0, 5), dtype=np.int64),
        "ids":    np.array(ids_train)
    }
    val = {
        "Xc":     pack(Xc_val, (0, context_len, 5)),
        "Y":      pack(Y_val,  (0, horizon)),
        "static": np.stack(S_val) if len(S_val) else np.zeros((0, 5), dtype=np.int64),
        "ids":    np.array(ids_val)
    }

    print(f"  - Windows ready: train={train['Xc'].shape[0]}, val={val['Xc'].shape[0]}")
    return train, val
