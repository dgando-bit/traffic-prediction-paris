from __future__ import annotations

from datetime import timedelta, date

import holidays
import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)

SCHOOL_HOLIDAY_PERIODS_ZONE_C = [
    (
        date(2025, 7, 5),
        date(2025, 8, 31),
    ),
]


def add_school_holiday_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add school-holiday features for Paris (Zone C).

    Dates are evaluated using Europe/Paris local time.
    """
    if "timestamp_paris" not in df.columns:
        raise ValueError(
            "Column 'timestamp_paris' is required "
            "to create school holiday features."
        )

    df = df.copy()

    local_dates = df["timestamp_paris"].dt.date

    def is_school_holiday(
        current_date: date,
    ) -> bool:
        return any(
            start_date
            <= current_date
            <= end_date
            for start_date, end_date
            in SCHOOL_HOLIDAY_PERIODS_ZONE_C
        )

    df["is_school_holiday"] = (
        local_dates.apply(
            is_school_holiday
        )
    )

    logger.info(
        "School holiday features created: "
        "%s holiday records / %s records",
        int(df["is_school_holiday"].sum()),
        len(df),
    )

    return df

def add_public_holiday_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add French public-holiday features.

    The local Paris date is used rather than UTC because
    holidays are calendar concepts in local time.
    """
    if "timestamp_paris" not in df.columns:
        raise ValueError(
            "Column 'timestamp_paris' is required "
            "to create calendar features."
        )

    df = df.copy()

    local_dates = df["timestamp_paris"].dt.date

    years = sorted(
        df["timestamp_paris"]
        .dropna()
        .dt.year
        .unique()
        .tolist()
    )

    if not years:
        raise ValueError(
            "No valid timestamp available "
            "to create calendar features."
        )

    french_holidays = holidays.France(
        years=years,
    )

    holiday_dates = set(
        french_holidays.keys()
    )

    df["is_public_holiday"] = (
        local_dates.isin(holiday_dates)
    )

    df["is_day_before_public_holiday"] = (
        local_dates.apply(
            lambda date: (
                date + timedelta(days=1)
                in holiday_dates
            )
        )
    )

    df["is_day_after_public_holiday"] = (
        local_dates.apply(
            lambda date: (
                date - timedelta(days=1)
                in holiday_dates
            )
        )
    )

    logger.info(
        "Public holiday features created: "
        "%s holidays / %s records",
        int(df["is_public_holiday"].sum()),
        len(df),
    )

    return df