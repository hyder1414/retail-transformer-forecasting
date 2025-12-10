from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    # time-series structure
    date_col: str = "Date"
    target_col: str = "Weekly_Sales"
    group_cols: List[str] = None
    input_length: int = 52
    output_length: int = 4
    min_history: int = 60  # min rows per (Store, Dept) to form windows

    # base feature columns in joined dataset
    # NOTE: we add Type_id, Store_id, Dept_id later in prepare_features
    base_feature_cols: List[str] = None


# default config used across the project
CONFIG = DataConfig(
    group_cols=["Store", "Dept"],
    input_length=52,
    output_length=4,
    min_history=60,
    base_feature_cols=[
        # past target as a feature
        "Weekly_Sales",
        # numeric covariates
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        # binary flags
        "IsHoliday",
        # markdowns
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        # store-level
        "Size",
        # calendar features
        "year",
        "month",
        "weekofyear",
        "dayofweek",
        # weather-derived feature
        "temp_anomaly",
    ],
)
