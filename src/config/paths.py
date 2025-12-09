from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_CSV = RAW_DIR / "train.csv"
FEATURES_CSV = RAW_DIR / "features.csv"
STORES_CSV = RAW_DIR / "stores.csv"
