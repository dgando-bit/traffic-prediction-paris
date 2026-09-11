import pandas as pd
from traffic_prediction.ingestion.events import (
    fetch_events,
)


def main() -> None:
    df = fetch_events()

    df["date_start"] = pd.to_datetime(
        df["date_start"],
        utc=True,
        errors="coerce",
    )

    df["date_end"] = pd.to_datetime(
        df["date_end"],
        utc=True,
        errors="coerce",
    )

    df["duration_hours"] = (
        df["date_end"]
        - df["date_start"]
    ).dt.total_seconds() / 3600

    print()
    print("=== Occurrences ===")

    print(
        "Events:",
        len(df),
    )

    print(
        "Occurrences non-null:",
        df["occurrences"].notna().sum(),
    )

    print()
    print("=== Event durations ===")

    print(
        "<= 24h:",
        (df["duration_hours"] <= 24).sum(),
    )

    print(
        "> 24h:",
        (df["duration_hours"] > 24).sum(),
    )

    print(
        "> 7 days:",
        (df["duration_hours"] > 24 * 7).sum(),
    )

    print(
        "> 30 days:",
        (df["duration_hours"] > 24 * 30).sum(),
    )

    print(
        "> 365 days:",
        (df["duration_hours"] > 24 * 365).sum(),
    )

    print()
    print("=== Examples with occurrences ===")

    columns = [
        "event_id",
        "title",
        "date_start",
        "date_end",
        "duration_hours",
        "occurrences",
    ]

    print(
        df[
            df["occurrences"].notna()
        ][columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=== Longest events ===")

    print(
        df[
            columns
        ]
        .sort_values(
            "duration_hours",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()