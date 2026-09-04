from traffic_prediction.pipelines.traffic_ingestion import (
    run_incremental_ingestion,
)


def main() -> None:
    inserted_rows = run_incremental_ingestion()

    print(
        f"Ingestion completed: "
        f"{inserted_rows} rows processed."
    )


if __name__ == "__main__":
    main()