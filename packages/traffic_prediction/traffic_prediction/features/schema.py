from __future__ import annotations

PREDICTION_HORIZONS = (
    1,
    2,
    3,
)


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


def get_target_column(
    horizon_hours: int,
) -> str:
    """
    Return the target column corresponding to a prediction horizon.

    Example:
        1 -> target_k_1h
        2 -> target_k_2h
        3 -> target_k_3h
    """
    if horizon_hours <= 0:
        raise ValueError(
            "horizon_hours must be greater than 0."
        )

    return f"target_k_{horizon_hours}h"


TARGET_COLUMNS = {
    horizon: get_target_column(
        horizon
    )
    for horizon in PREDICTION_HORIZONS
}


# Backward compatibility with the existing +1h pipeline.
TARGET_COLUMN = TARGET_COLUMNS[1]