from __future__ import annotations

import numpy as np
import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)

EVENT_RADIUS_M = 5000.0
EARTH_RADIUS_M = 6_371_000.0


def _haversine_distances_m(
    latitude: float,
    longitude: float,
    event_latitudes: np.ndarray,
    event_longitudes: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance between one road location
    and multiple event locations.
    """

    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)

    lat2 = np.radians(event_latitudes)
    lon2 = np.radians(event_longitudes)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(
        np.sqrt(a)
    )

    return EARTH_RADIUS_M * c


def add_event_features(
    traffic_df: pd.DataFrame,
    event_occurrences_df: pd.DataFrame,
    radius_m: float = EVENT_RADIUS_M,
    horizon_hours: int = 1,
) -> pd.DataFrame:
    """
    Add event features around each traffic sensor.

    Features are aligned with the prediction target time:

        target_time = timestamp_utc + horizon_hours

    An event is considered active when:

        occurrence_start <= target_time <= occurrence_end

    Only events located within `radius_m` of the road sensor
    are considered.
    """

    required_traffic_columns = {
        "iu_ac",
        "timestamp_utc",
        "latitude",
        "longitude",
    }

    missing = (
        required_traffic_columns
        - set(traffic_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing traffic columns: {sorted(missing)}"
        )

    required_event_columns = {
        "occurrence_start",
        "occurrence_end",
        "latitude",
        "longitude",
    }

    missing = (
        required_event_columns
        - set(event_occurrences_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing event columns: {sorted(missing)}"
        )

    result = traffic_df.copy()

    result["timestamp_utc"] = pd.to_datetime(
        result["timestamp_utc"],
        utc=True,
    )

    events = event_occurrences_df.copy()

    events["occurrence_start"] = pd.to_datetime(
        events["occurrence_start"],
        utc=True,
    )

    events["occurrence_end"] = pd.to_datetime(
        events["occurrence_end"],
        utc=True,
    )

    events = events.dropna(
        subset=[
            "occurrence_start",
            "occurrence_end",
            "latitude",
            "longitude",
        ]
    ).copy()

    event_latitudes = (
        events["latitude"]
        .to_numpy(dtype=float)
    )

    event_longitudes = (
        events["longitude"]
        .to_numpy(dtype=float)
    )

    event_counts = np.zeros(
        len(result),
        dtype=np.int32,
    )

    target_timestamps = (
        result["timestamp_utc"]
        + pd.Timedelta(
            hours=horizon_hours
        )
    )

    logger.info(
        "Building event features for %s records "
        "with radius %.1f km",
        len(result),
        radius_m / 1000.0,
    )

    # Only ~20 roads in the current dataset.
    # We compute road -> event distances once per road,
    # instead of once per traffic row.
    for iu_ac, group in result.groupby(
        "iu_ac",
        sort=False,
    ):
        valid_coordinates = group[
            ["latitude", "longitude"]
        ].dropna()

        if valid_coordinates.empty:
            logger.warning(
                "No coordinates for road %s",
                iu_ac,
            )
            continue

        road_latitude = float(
            valid_coordinates.iloc[0]["latitude"]
        )

        road_longitude = float(
            valid_coordinates.iloc[0]["longitude"]
        )

        distances = _haversine_distances_m(
            latitude=road_latitude,
            longitude=road_longitude,
            event_latitudes=event_latitudes,
            event_longitudes=event_longitudes,
        )

        nearby_mask = (
            distances <= radius_m
        )

        nearby_events = events.loc[
            nearby_mask
        ]

        if nearby_events.empty:
            continue

        starts = np.sort(
            nearby_events[
                "occurrence_start"
            ].array.asi8
        )

        ends = np.sort(
            nearby_events[
                "occurrence_end"
            ].array.asi8
        )

        group_indices = group.index

        target_ns = (
            target_timestamps.loc[
                group_indices
            ]
            .array.asi8
        )

        # Number of intervals where:
        #
        # start <= target
        # end >= target
        #
        # active =
        # starts_before_target
        # - ends_before_target
        starts_before = np.searchsorted(
            starts,
            target_ns,
            side="right",
        )

        ends_before = np.searchsorted(
            ends,
            target_ns,
            side="left",
        )

        counts = (
            starts_before
            - ends_before
        )

        # group.index contains original dataframe indexes.
        # Convert them to positional indexes safely.
        positions = result.index.get_indexer(
            group_indices
        )

        event_counts[
            positions
        ] = counts

    result[
        "event_count_nearby_target_1h"
    ] = event_counts

    result[
        "has_event_nearby_target_1h"
    ] = (
        result[
            "event_count_nearby_target_1h"
        ] > 0
    ).astype("int8")

    logger.info(
        "Event features created: %s records / "
        "%s with nearby events",
        len(result),
        int(
            result[
                "has_event_nearby_target_1h"
            ].sum()
        ),
    )

    return result