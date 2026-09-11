from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient
from shared.config import get_settings

from traffic_prediction.features.road_features import (
    add_road_features,
)
from traffic_prediction.features.schema import (
    FEATURE_COLUMNS,
    PREDICTION_HORIZONS,
)
from traffic_prediction.features.time_features import (
    add_time_features,
)
from traffic_prediction.features.traffic_features import (
    add_traffic_lag_features,
)
from traffic_prediction.processing.cleaning import (
    clean_traffic_data,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_recent_traffic_dataframe,
    insert_predictions,
)

settings = get_settings()


ROAD_REFERENCE_PATH = Path(
    "data/raw/reference/road_reference.parquet"
)


def get_registered_model_name(
    horizon_hours: int,
) -> str:
    """
    Return the MLflow registered-model name for
    a prediction horizon.
    """
    if horizon_hours <= 0:
        raise ValueError(
            "horizon_hours must be greater than 0."
        )

    return (
        f"{settings.registered_model_name}"
        f"-{horizon_hours}h"
    )


def load_champion_model(
    horizon_hours: int,
) -> tuple[object, str]:
    """
    Load the champion model for one prediction horizon.

    Returns
    -------
    tuple
        Loaded model and concrete MLflow model version.
    """
    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    client = MlflowClient()

    registered_model_name = (
        get_registered_model_name(
            horizon_hours
        )
    )

    model_version = (
        client.get_model_version_by_alias(
            registered_model_name,
            settings.api_model_alias,
        )
    )

    model_uri = (
        f"models:/"
        f"{registered_model_name}"
        f"@{settings.api_model_alias}"
    )

    print(
        f"Loading model "
        f"{registered_model_name}"
        f"@{settings.api_model_alias} "
        f"(version {model_version.version})"
    )

    model = mlflow.lightgbm.load_model(
        model_uri
    )

    return (
        model,
        str(model_version.version),
    )


def build_prediction_features(
    *,
    history_hours: int = 48,
    road_reference_path: str | Path = (
        ROAD_REFERENCE_PATH
    ),
) -> pd.DataFrame:
    """
    Build the features shared by all prediction
    horizons.
    """
    road_reference_path = Path(
        road_reference_path
    )

    if not road_reference_path.exists():
        raise FileNotFoundError(
            f"Road reference file not found: "
            f"{road_reference_path}"
        )

    with get_db_session() as session:
        traffic_df = (
            get_recent_traffic_dataframe(
                session,
                hours=history_hours,
            )
        )

    if traffic_df.empty:
        raise ValueError(
            "No traffic observations available "
            "in PostgreSQL."
        )

    print(
        f"Loaded {len(traffic_df)} traffic "
        f"observations from PostgreSQL"
    )

    road_reference_df = pd.read_parquet(
        road_reference_path
    )

    features_df = clean_traffic_data(
        traffic_df
    )

    features_df = add_time_features(
        features_df
    )

    features_df = add_road_features(
        features_df,
        road_reference_df,
    )

    eligible_road_ids = set(
        road_reference_df["iu_ac"]
        .astype(str)
    )

    features_df = features_df[
        features_df["iu_ac"]
        .astype(str)
        .isin(eligible_road_ids)
    ].copy()

    print(
        f"Kept "
        f"{features_df['iu_ac'].nunique()} "
        "eligible roads after "
        "road-reference filtering"
    )

    features_df = add_traffic_lag_features(
        features_df,
        lags=(1, 2, 24),
    )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing prediction features: "
            f"{missing_features}"
        )

    features_df = (
        features_df
        .sort_values(
            [
                "iu_ac",
                "timestamp_utc",
            ]
        )
        .reset_index(drop=True)
    )

    return features_df


def select_latest_observations(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep the most recent observation available
    for each road.
    """
    if features_df.empty:
        return features_df.copy()

    latest_df = (
        features_df
        .sort_values(
            [
                "iu_ac",
                "timestamp_utc",
            ]
        )
        .groupby(
            "iu_ac",
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_df


def build_horizon_predictions(
    prediction_df: pd.DataFrame,
    *,
    horizon_hours: int,
) -> list[dict]:
    """
    Generate prediction records for one horizon.
    """
    model, model_version = (
        load_champion_model(
            horizon_hours
        )
    )

    horizon_df = prediction_df.copy()

    horizon_df["predicted_k"] = (
        model.predict(
            horizon_df[
                FEATURE_COLUMNS
            ]
        )
    )

    horizon_df[
        "prediction_timestamp_utc"
    ] = horizon_df[
        "timestamp_utc"
    ]

    horizon_df[
        "target_timestamp_utc"
    ] = (
        horizon_df[
            "timestamp_utc"
        ]
        + pd.Timedelta(
            hours=horizon_hours
        )
    )

    horizon_df[
        "horizon_hours"
    ] = horizon_hours

    horizon_df[
        "model_version"
    ] = model_version

    records = (
        horizon_df[
            [
                "iu_ac",
                "prediction_timestamp_utc",
                "target_timestamp_utc",
                "horizon_hours",
                "predicted_k",
                "model_version",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    print(
        f"+{horizon_hours}h: "
        f"{len(records)} predictions "
        f"generated using model "
        f"version {model_version}"
    )

    return records


def run_predictions(
    *,
    history_hours: int = 48,
) -> int:
    """
    Generate predictions for every configured
    horizon and store them in PostgreSQL.

    Returns
    -------
    int
        Total number of predictions processed.
    """
    print(
        "Building prediction features "
        f"(history_hours={history_hours})"
    )

    features_df = (
        build_prediction_features(
            history_hours=history_hours,
        )
    )

    prediction_df = (
        select_latest_observations(
            features_df
        )
    )

    if prediction_df.empty:
        print(
            "No observations available "
            "for prediction."
        )
        return 0

    print(
        f"Generating predictions for "
        f"{len(prediction_df)} roads "
        f"and horizons "
        f"{PREDICTION_HORIZONS}"
    )

    records: list[dict] = []

    for horizon_hours in (
        PREDICTION_HORIZONS
    ):
        horizon_records = (
            build_horizon_predictions(
                prediction_df,
                horizon_hours=(
                    horizon_hours
                ),
            )
        )

        records.extend(
            horizon_records
        )

    with get_db_session() as session:
        insert_predictions(
            session,
            records,
        )

    processed = len(records)

    print(
        f"{processed} predictions stored "
        f"across "
        f"{len(PREDICTION_HORIZONS)} "
        "horizons"
    )

    return processed