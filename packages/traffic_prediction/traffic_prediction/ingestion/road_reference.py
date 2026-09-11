from __future__ import annotations

import json
from collections.abc import Sequence
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import httpx
import pandas as pd
from shared.logging import get_logger

logger = get_logger(__name__)


ROAD_REFERENCE_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/"
    "datasets/referentiel-comptages-routiers/records"
)


def _haversine_distance_m(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """
    Compute the great-circle distance between two geographic points.

    Parameters
    ----------
    lon1, lat1:
        Longitude and latitude of the first point.

    lon2, lat2:
        Longitude and latitude of the second point.

    Returns
    -------
    float
        Distance in meters.
    """
    earth_radius_m = 6_371_000

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = (
        sin(delta_phi / 2) ** 2
        + cos(phi1)
        * cos(phi2)
        * sin(delta_lambda / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return earth_radius_m * c


def _compute_linestring_length_m(
    geo_shape: dict,
) -> float | None:
    """
    Compute the total length of a GeoJSON LineString.

    The length is calculated by summing the Haversine distance
    between each consecutive pair of coordinates.

    Parameters
    ----------
    geo_shape:
        GeoJSON Feature containing a LineString geometry.

    Returns
    -------
    float | None
        Total road segment length in meters.
    """
    if not isinstance(geo_shape, dict):
        return None

    geometry = geo_shape.get("geometry")

    if not geometry:
        return None

    if geometry.get("type") != "LineString":
        return None

    coordinates = geometry.get("coordinates", [])

    if len(coordinates) < 2:
        return None

    total_length = 0.0

    for start, end in zip(
            coordinates[:-1],
            coordinates[1:],
            strict=True,
    ):
        lon1, lat1 = start
        lon2, lat2 = end

        total_length += _haversine_distance_m(
            lon1,
            lat1,
            lon2,
            lat2,
        )

    return total_length


def fetch_road_reference(
    road_ids: Sequence[str],
    *,
    batch_size: int = 50,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Fetch road reference information from Paris Open Data.

    Road identifiers are fetched in batches to avoid making one
    HTTP request per road.

    Parameters
    ----------
    road_ids:
        Road segment identifiers (iu_ac).

    batch_size:
        Number of road IDs included in one API filter.

    page_size:
        Number of records requested per API page.

    Returns
    -------
    pd.DataFrame
        Raw road reference records.
    """
    road_ids = list(
        dict.fromkeys(
            str(road_id)
            for road_id in road_ids
        )
    )

    if not road_ids:
        return pd.DataFrame()

    rows: list[dict] = []

    batches = [
        road_ids[index:index + batch_size]
        for index in range(
            0,
            len(road_ids),
            batch_size,
        )
    ]

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):
        road_filter = " OR ".join(
            f'iu_ac="{road_id}"'
            for road_id in batch
        )

        where_clause = f"({road_filter})"

        offset = 0

        while True:
            response = httpx.get(
                ROAD_REFERENCE_URL,
                params={
                    "where": where_clause,
                    "limit": page_size,
                    "offset": offset,
                },
                timeout=30,
            )

            response.raise_for_status()

            payload = response.json()

            page = payload.get(
                "results",
                [],
            )

            if not page:
                break

            rows.extend(page)

            if len(page) < page_size:
                break

            offset += len(page)

        logger.info(
            "Road reference batch %s/%s fetched "
            "(%s roads, %s total records)",
            batch_index,
            len(batches),
            len(batch),
            len(rows),
        )

    logger.info(
        "Fetched %s road reference records",
        len(rows),
    )

    return pd.DataFrame(rows)


def normalize_road_reference(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Paris road reference data.

    Operations performed:
    - convert iu_ac to string
    - parse validity dates
    - keep the most recent version of each road
    - extract latitude and longitude
    - calculate road segment length
    - serialize GeoJSON columns for Parquet storage

    Parameters
    ----------
    df:
        Raw road reference dataframe.

    Returns
    -------
    pd.DataFrame
        Normalized road reference dataframe.
    """
    if df.empty:
        logger.warning(
            "Road reference dataframe is empty"
        )
        return df.copy()

    result = df.copy()

    # ---------------------------------------------------------
    # Types
    # ---------------------------------------------------------

    result["iu_ac"] = result["iu_ac"].astype(
        "string"
    )

    result["date_debut"] = pd.to_datetime(
        result["date_debut"],
        utc=True,
        errors="coerce",
    )

    result["date_fin"] = pd.to_datetime(
        result["date_fin"],
        utc=True,
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Reference version
    # ---------------------------------------------------------
    # Some road segments have several historical versions.
    # For V2 we keep the most recent available version.

    result = (
        result.sort_values("date_debut")
        .drop_duplicates(
            subset=["iu_ac"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Geographic coordinates
    # ---------------------------------------------------------

    result["longitude"] = result[
        "geo_point_2d"
    ].apply(
        lambda value: value.get("lon")
        if isinstance(value, dict)
        else None
    )

    result["latitude"] = result[
        "geo_point_2d"
    ].apply(
        lambda value: value.get("lat")
        if isinstance(value, dict)
        else None
    )

    # ---------------------------------------------------------
    # Road length
    # ---------------------------------------------------------
    # Must be calculated before serializing geo_shape.

    result["road_length_m"] = result[
        "geo_shape"
    ].apply(
        _compute_linestring_length_m
    )

    # ---------------------------------------------------------
    # Serialize complex columns
    # ---------------------------------------------------------
    # PyArrow / Parquet can have issues with heterogeneous
    # nested dictionaries, so we store GeoJSON as JSON strings.

    result["geo_point_2d"] = result[
        "geo_point_2d"
    ].apply(
        lambda value: json.dumps(value)
        if isinstance(value, dict)
        else None
    )

    result["geo_shape"] = result[
        "geo_shape"
    ].apply(
        lambda value: json.dumps(value)
        if isinstance(value, dict)
        else None
    )

    # ---------------------------------------------------------
    # Final ordering
    # ---------------------------------------------------------

    result = result.sort_values(
        "iu_ac"
    ).reset_index(drop=True)

    logger.info(
        "Normalized road reference: %s roads",
        len(result),
    )

    return result


def save_road_reference(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save normalized road reference data to Parquet.

    Parameters
    ----------
    df:
        Normalized road reference dataframe.

    output_path:
        Destination Parquet path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Road reference saved to %s",
        output_path,
    )