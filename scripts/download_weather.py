from pathlib import Path

from traffic_prediction.ingestion.weather import (
    ingest_historical_weather,
)


OUTPUT_PATH = Path(
    "data/raw/weather/weather_paris_historical.parquet"
)

START_DATE = "2025-07-01"
END_DATE = "2026-06-16"


def main() -> None:
    df = ingest_historical_weather(
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=OUTPUT_PATH,
    )

    print()
    print("=== Historical weather ===")

    print(f"Rows: {len(df)}")

    print(
        "Period:",
        df["timestamp_utc"].min(),
        "->",
        df["timestamp_utc"].max(),
    )

    print()
    print("Missing values:")
    print(df.isna().sum())

    print()
    print(df.head())

    assert len(df) > 0

    assert (
        df["timestamp_utc"]
        .is_monotonic_increasing
    )

    assert (
        df["timestamp_utc"]
        .duplicated()
        .sum()
        == 0
    )

    print()
    print(
        "✅ Historical weather ingestion OK"
    )


if __name__ == "__main__":
    main()