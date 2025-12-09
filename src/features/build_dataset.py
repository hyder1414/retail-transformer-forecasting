import pandas as pd

from src.config.paths import TRAIN_CSV, FEATURES_CSV, STORES_CSV, INTERIM_DIR


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df["Date"].dt.weekday  # 0=Mon
    return df


def add_weather_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    if "Temperature" not in df.columns:
        return df

    grp_mean = df.groupby(["Store", "weekofyear"])["Temperature"].transform("mean")
    df["temp_anomaly"] = df["Temperature"] - grp_mean
    return df


def main():
    print("Reading raw CSVs...")
    train = pd.read_csv(TRAIN_CSV)
    features = pd.read_csv(FEATURES_CSV)
    stores = pd.read_csv(STORES_CSV)

    print("Shapes:")
    print("  train:", train.shape)
    print("  features:", features.shape)
    print("  stores:", stores.shape)

    print("Merging train + features...")
    df = pd.merge(
        train,
        features,
        on=["Store", "Date"],
        how="left",
        suffixes=("", "_feat"),
    )

    print("Merging with stores metadata...")
    df = pd.merge(df, stores, on="Store", how="left")

    print("Adding calendar features...")
    df = add_calendar_features(df)

    print("Adding weather anomaly feature...")
    df = add_weather_anomaly(df)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "joined.parquet"

    print(f"Saving joined dataset to: {out_path}")
    df.to_parquet(out_path, index=False)

    print("Done. Final shape:", df.shape)
    print("Columns:", sorted(df.columns))


if __name__ == "__main__":
    main()
