# data preprocessing.py
"""
preprocessing for Walmart store sales transformer model.
Main entry point:
    prepare_walmart_data(
        seq_len=16,
        horizon=4,
        data_dir="cache",
        force_recompute=False)

Output:
    X: np.ndarray, shape (num_samples, seq_len, num_features)
    y: np.ndarray, shape (num_samples, horizon)
    scaler: fitted sklearn StandardScaler for continuous features
    feature_columns: list of feature names in the same order as X 
"""
from pathlib import Path
import numpy as np
import joblib
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import pandas as pd
#Load  CSVs
def _load_raw_walmart_data(base_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Checking for cached data
    base_dir = Path(base_dir)
    train_path = base_dir / "train.csv"
    features_path = base_dir / "features.csv"
    stores_path = base_dir / "stores.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {base_dir}")
    if not features_path.exists():
        raise FileNotFoundError(f"features.csv not found in {base_dir}")
    if not stores_path.exists():
        raise FileNotFoundError(f"stores.csv not found in {base_dir}")

    train_df = pd.read_csv(train_path)
    features_df = pd.read_csv(features_path)
    stores_df = pd.read_csv(stores_path)

    # Ensure Date is datetime
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    features_df["Date"] = pd.to_datetime(features_df["Date"])

    return train_df, features_df, stores_df
#Merging the datasets
def _merge_data(train_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:

    # Merge store info
    df = train_df.merge(stores_df, on="Store", how="left")
    df = df.merge(features_df, on=["Store", "Date"], how="left")

    #Since there is duplicate on IsHoliday column for both csv, it returns 2 same column after merging
    if "IsHoliday_x" in df.columns:
        df["IsHoliday"] = df["IsHoliday_x"]
    elif "IsHoliday_y" in df.columns:
        df["IsHoliday"] = df["IsHoliday_y"]
    elif "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"]
    else:
        df["IsHoliday"] = 0
    # Making sure holiday is 0/1
    df["IsHoliday"] = df["IsHoliday"].fillna(0).astype(int)
    for col in ["IsHoliday_x", "IsHoliday_y"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df



## Feature engineering and scaling
def _engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    After checking for nan values, fill missing values, add time features, encode categories, and scale continuous features.
    Returns the transformed df, list of feature columns, and fitted scaler.
    """

    # log1p of weekly sales for scaling
    if (df["Weekly_Sales"] < 0).any():
        df["Weekly_Sales_Clipped"] = np.clip(df["Weekly_Sales"], a_min=0, a_max=None)
        source_col = "Weekly_Sales_Clipped"
    else:
        source_col = "Weekly_Sales"

    df["Weekly_Sales_Log"] = np.log1p(df[source_col])

    # Ensure MarkDown columns nans
    for i in range(1, 6):
        col = f"MarkDown{i}"
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Filling numeric columns per Store, then with global median
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = (df.groupby("Store")[col].transform(lambda x: x.ffill().bfill())
                       .fillna(df[col].median()))
        else:
            df[col] = 0.0

    # Time features engineering
    """
    By creating time features like day of year and week of year, it will let the model 
    learn more about time trend and seasonality
    """
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear

    if "Type" in df.columns:
        df["Type"] = df["Type"].map({"A": 0, "B": 1, "C": 2}).fillna(-1).astype(int)
    else:
        df["Type"] = -1

    # Sorting by entity + time
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    # Final features  for the model
    feature_columns = [
        "Store", "Dept", "Type", "Size",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "IsHoliday", "Year", "Month", "WeekOfYear", "DayOfYear",
    ]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    ## Scaling
    # Continuous columns to be scaled
    cont_cols = ["Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "WeekOfYear", "DayOfYear",]
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    return df, feature_columns, scaler



# Building sequences per store, Dept

def _build_sequences(df: pd.DataFrame, feature_columns: List[str], seq_len: int, horizon: int, ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y sequences per Store, Dept. With sequence modeling like last 4 months, 
    the model could have more attention and gain insight on timely trend
    X: num_samples, seq_len, num_features
    y: num_samples, horizon (note: l;og sclaed, thus the error doesn't define usd)
    """

    def create_seq(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_vals = group[feature_columns].values.astype(np.float32)
        y_vals = group["Weekly_Sales_Log"].values.astype(np.float32)

        seqs_x, seqs_y = [], []
        max_start = len(group) - seq_len - horizon + 1
        for start in range(max_start):
            end = start + seq_len
            out_end = end + horizon
            seqs_x.append(X_vals[start:end])
            seqs_y.append(y_vals[end:out_end])
        if not seqs_x:
            return (
                np.empty((0, seq_len, len(feature_columns)), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32),
            )
        return np.stack(seqs_x), np.stack(seqs_y)
    X_list, y_list = [], []
    for (_, _), g in df.groupby(["Store", "Dept"], sort=False):
        if len(g) >= seq_len + horizon:
            xs, ys = create_seq(g)
            if xs.shape[0] > 0:
                X_list.append(xs)
                y_list.append(ys)

    if len(X_list) == 0:
        raise ValueError("Couldn't create seq, check  input parameters")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def prepare_walmart_data(
    seq_len: int = 16,
    horizon: int = 4,
    data_dir: str = "cache",
    force_recompute: bool = False,
    base_dir: str = ".",) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    X_path = data_dir / f"X_seq{seq_len}_h{horizon}.npy"
    y_path = data_dir / f"y_seq{seq_len}_h{horizon}.npy"
    scaler_path = data_dir / f"scaler_seq{seq_len}_h{horizon}.pkl"
    cols_path = data_dir / f"feature_columns_seq{seq_len}_h{horizon}.pkl"

    # Load from cache if available
    if (
        not force_recompute
        and X_path.exists()
        and y_path.exists()
        and scaler_path.exists()
        and cols_path.exists()):
        print(f"Loading cached data")
        X = np.load(X_path)
        y = np.load(y_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(cols_path)
        print(f"Loaded {X.shape[0]:,} samples with sequence length {seq_len} and horizon {horizon}.")
        return X, y, scaler, feature_columns

   #if not, 
    print("Loading raw CSV files")
    train_df, features_df, stores_df = _load_raw_walmart_data(base_dir=base_dir)
    df = _merge_data(train_df, features_df, stores_df)
    df, feature_columns, scaler = _engineer_features(df)
    X, y = _build_sequences(df, feature_columns, seq_len=seq_len, horizon=horizon)

    # Save Cache again
    np.save(X_path, X)
    np.save(y_path, y)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, cols_path)

    print(f" output:{X.shape[0]:,} samples. Saved to: {data_dir}")
    return X, y, scaler, feature_columns



#Test Run
if __name__ == "__main__":
    X, y, scaler, feature_cols = prepare_walmart_data(
        seq_len=16,
        horizon=4,
        data_dir="cache",
        force_recompute=True,
        base_dir=".")
    print("Shapes:")
    print("  X:", X.shape) #expected output: (361000,16,18)
    print("  y:", y.shape) #Expected: (361000, 4)
    print("Num features:", len(feature_cols))
