from traffic_prediction.inference.predictor import (
    run_predictions,
)


def main() -> None:
    df = run_predictions()

    print()
    print("=== Predictions completed ===")

    print(
        df[
            [
                "iu_ac",
                "k",
                "predicted_k",
            ]
        ]
        .sort_values(
            "predicted_k",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print(
        f"Saved {len(df)} predictions "
        "to PostgreSQL"
    )


if __name__ == "__main__":
    main()