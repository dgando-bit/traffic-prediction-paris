from pathlib import Path

from traffic_prediction.ingestion.events import (
    build_event_occurrences,
    ingest_events,
)

OUTPUT_PATH = Path(
    "data/raw/events/events_paris.parquet"
)


def main() -> None:
    df = ingest_events(
        OUTPUT_PATH
    )

    occurrences_df = build_event_occurrences(
        df
    )

    OCCURRENCES_PATH = Path(
        "data/processed/event_occurrences.parquet"
    )

    OCCURRENCES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    occurrences_df.to_parquet(
        OCCURRENCES_PATH,
        index=False,
    )

    print()
    print("=== Event occurrences ===")

    print(
        f"Occurrences: {len(occurrences_df)}"
    )

    print(
        "Period:",
        occurrences_df["occurrence_start"].min(),
        "->",
        occurrences_df["occurrence_end"].max(),
    )

    print(
        f"Unique events: "
        f"{occurrences_df['event_id'].nunique()}"
    )

    print()
    print(
        f"Occurrences saved: {OCCURRENCES_PATH}"
    )

    print()
    print("=== Events dataset ===")

    print(
        f"Rows: {len(df)}"
    )

    print(
        "Period:",
        df["date_start"].min(),
        "->",
        df["date_end"].max(),
    )

    print()
    print("Missing values:")

    print(
        df[
            [
                "date_start",
                "date_end",
                "latitude",
                "longitude",
            ]
        ].isna().sum()
    )

    print()
    print("Events with coordinates:")

    print(
        df[
            ["latitude", "longitude"]
        ]
        .notna()
        .all(axis=1)
        .sum()
    )

    print()
    print(df.head(10))

    print()
    print(
        f"Dataset saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()