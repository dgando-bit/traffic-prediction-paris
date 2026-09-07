from __future__ import annotations

TARGET_COLUMN = "target_k_1h"

FEATURE_COLUMNS = [
    "q",
    "k",
    "hour",
    "day_of_week",
    "is_weekend",
    "q_lag_1h",
    "k_lag_1h",
    "q_lag_2h",
    "k_lag_2h",
    "q_lag_24h",
    "k_lag_24h",
    "latitude",
    "longitude",
    "road_length_m",
]