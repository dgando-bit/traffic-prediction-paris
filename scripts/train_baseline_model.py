import pandas as pd

from traffic_prediction.features.ml_dataset import (
    TARGET_COLUMN,
)
from traffic_prediction.models.baseline import (
    evaluate_persistence_baseline,
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import (
    evaluate_regression,
)
from traffic_prediction.models.train import (
    train_hist_gradient_boosting,
)


INPUT_PATH = (
    "data/processed/ml_dataset_multi_20.parquet"
)


def main() -> None:
    # ---------------------------------------------------------------
    # Load ML dataset
    # ---------------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    print("=== ML Dataset ===")
    print(f"Nombre de lignes : {len(df)}")

    print(
        "Période :",
        df["timestamp_utc"].min(),
        "->",
        df["timestamp_utc"].max(),
    )

    # ---------------------------------------------------------------
    # Temporal split
    # ---------------------------------------------------------------

    train_df, test_df = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    print()
    print("=== Temporal split ===")

    print(
        f"Train : {len(train_df)} lignes"
    )

    print(
        f"Test  : {len(test_df)} lignes"
    )

    # ---------------------------------------------------------------
    # Common evaluation dataset
    # ---------------------------------------------------------------

    evaluation_df = test_df[
        test_df[TARGET_COLUMN].notna()
        & test_df["k"].notna()
        ].copy()

    print()
    print(
        f"Common evaluation samples : {len(evaluation_df)}"
    )

    # ---------------------------------------------------------------
    # Persistence baseline
    # ---------------------------------------------------------------

    baseline_metrics = (
        evaluate_persistence_baseline(
            evaluation_df
        )
    )

    # ---------------------------------------------------------------
    # Train ML model
    # ---------------------------------------------------------------

    training_result = (
        train_hist_gradient_boosting(
            train_df
        )
    )

    model = training_result.model

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    X_test = evaluation_df[
        training_result.feature_columns
    ]

    y_test = evaluation_df[
        TARGET_COLUMN
    ]

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------------
    # ML evaluation
    # ---------------------------------------------------------------

    ml_metrics = evaluate_regression(
        y_true=y_test.to_numpy(),
        y_pred=predictions,
    )

    # ---------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------

    print()
    print("=== Model comparison ===")

    print(
        f"{'Model':<25}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
    )

    print("-" * 49)

    print(
        f"{'Persistence':<25}"
        f"{baseline_metrics.mae:>12.4f}"
        f"{baseline_metrics.rmse:>12.4f}"
    )

    print(
        f"{'HistGradientBoosting':<25}"
        f"{ml_metrics.mae:>12.4f}"
        f"{ml_metrics.rmse:>12.4f}"
    )

    # ---------------------------------------------------------------
    # Improvement
    # ---------------------------------------------------------------

    mae_improvement = (
        (
            baseline_metrics.mae
            - ml_metrics.mae
        )
        / baseline_metrics.mae
        * 100
    )

    rmse_improvement = (
        (
            baseline_metrics.rmse
            - ml_metrics.rmse
        )
        / baseline_metrics.rmse
        * 100
    )

    print()
    print("=== Improvement vs persistence ===")

    print(
        f"MAE  : {mae_improvement:+.2f}%"
    )

    print(
        f"RMSE : {rmse_improvement:+.2f}%"
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print()
    print("=== Contrôles ===")

    assert ml_metrics.n_samples > 0

    assert ml_metrics.mae >= 0
    assert ml_metrics.rmse >= 0

    assert baseline_metrics.n_samples == ml_metrics.n_samples

    print("✅ ML training test OK")


if __name__ == "__main__":
    main()