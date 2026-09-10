from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient

from shared.config import get_settings
from traffic_prediction.features.road_features import add_road_features
from traffic_prediction.features.time_features import add_time_features
from traffic_prediction.features.traffic_features import add_traffic_lag_features
from traffic_prediction.storage.database import get_db_session
from traffic_prediction.storage.repositories import (
    get_recent_traffic_dataframe,
    insert_predictions,
)
from traffic_prediction.processing.cleaning import clean_traffic_data
from traffic_prediction.features.schema import FEATURE_COLUMNS

settings = get_settings()


ROAD_REFERENCE_PATH = Path(
    "data/raw/reference/road_reference.parquet"
)


def load_champion_model() -> tuple[object, str]:
    """
    Load the production model from the MLflow Model Registry.

    Returns
    -------
    tuple
        Loaded model and concrete MLflow model version.
    """
    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    client = MlflowClient()

    model_version = (
        client.get_model_version_by_alias(
            settings.registered_model_name,
            settings.api_model_alias,
        )
    )

    model_uri = (
        f"models:/"
        f"{settings.registered_model_name}"
        f"@{settings.api_model_alias}"
    )

    print(
        f"Loading model "
        f"{settings.registered_model_name}"
        f"@{settings.api_model_alias} "
        f"(version {model_version.version})"
    )

    model = mlflow.lightgbm.load_model(
        model_uri
    )

    return model, str(model_version.version)


def build_prediction_features(
    *,
    history_hours: int = 48,
    road_reference_path: str | Path = ROAD_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Build features used for one-hour-ahead traffic prediction.
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
        f"Kept {features_df['iu_ac'].nunique()} "
        "eligible roads after road-reference filtering"
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


def run_predictions(
    *,
    history_hours: int = 48,
) -> int:
    """
    Generate one-hour-ahead predictions using
    the MLflow champion model and store them
    in PostgreSQL.

    Returns
    -------
    int
        Number of predictions processed.
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

    model, model_version = (
        load_champion_model()
    )

    print(
        f"Generating predictions for "
        f"{len(prediction_df)} roads"
    )

    prediction_df = prediction_df.copy()

    prediction_df["predicted_k"] = (
        model.predict(
            prediction_df[
                FEATURE_COLUMNS
            ]
        )
    )

    prediction_df[
        "prediction_timestamp_utc"
    ] = prediction_df[
        "timestamp_utc"
    ]

    prediction_df[
        "target_timestamp_utc"
    ] = (
        prediction_df[
            "timestamp_utc"
        ]
        + pd.Timedelta(hours=1)
    )

    prediction_records = (
        prediction_df[
            [
                "iu_ac",
                "prediction_timestamp_utc",
                "target_timestamp_utc",
                "predicted_k",
            ]
        ]
        .copy()
    )

    prediction_records[
        "model_version"
    ] = model_version

    records = prediction_records.to_dict(
        orient="records"
    )

    with get_db_session() as session:
        insert_predictions(
            session,
            records,
        )

    processed = len(records)

    print(
        f"{processed} predictions stored "
        f"using model version "
        f"{model_version}"
    )

    return processed