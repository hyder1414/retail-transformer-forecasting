#This just centralizes “what is time / id / target / features” and the input_length / output_length

from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    time_col: str = "Date"
    id_cols: List[str] = ("Store", "Dept")
    target_col: str = "Weekly_Sales"

    # sequence lengths
    input_length: int = 52   # use 52 weeks of history
    output_length: int = 4   # predict next 4 weeks

    # which features go into the model
    # we'll filter to only existing cols later
    base_feature_cols: List[str] = (
        "Weekly_Sales",    # past target (will be normalized)
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "IsHoliday",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "Size",
        # "Type",  # <- leave this commented or removed
        "year",
        "month",
        "weekofyear",
        "dayofweek",
        "temp_anomaly",
    )




CONFIG = DataConfig()
