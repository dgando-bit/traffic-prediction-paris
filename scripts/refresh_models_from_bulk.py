from __future__ import annotations

import shutil
from pathlib import Path

from traffic_prediction.pipelines.dataset import (
    make_training_dataset,
)
from traffic_prediction.pipelines.features import (
    build_training_features,
)
from traffic_prediction.pipelines.promotion import (
    promote_all_candidates,
)
from traffic_prediction.pipelines.traffic_bulk_import import (
    run_bulk_import,
)
from traffic_prediction.pipelines.training import (
    train_and_register_all_horizons,
)


INBOX_DIR = Path(
    "data/raw/traffic/bulk/inbox"
)

ARCHIVE_DIR = Path(
    "data/raw/traffic/bulk/archive"
)

TRAINING_HISTORY_DAYS = 90


def get_pending_files() -> list[Path]:
    return sorted(
        INBOX_DIR.glob("*.parquet")
    )


def archive_files(
    files: list[Path],
) -> None:
    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in files:
        destination = (
            ARCHIVE_DIR / path.name
        )

        if destination.exists():
            print(
                f"Already archived: "
                f"{path.name}"
            )

            path.unlink()

            continue

        shutil.move(
            str(path),
            str(destination),
        )

        print(
            f"Archived: {path.name}"
        )

def main() -> None:
    pending_files = (
        get_pending_files()
    )

    if not pending_files:
        print(
            "No new Parquet files found "
            "in the bulk inbox."
        )
        return

    print()
    print("=" * 64)
    print("TRAFFIC MODEL REFRESH")
    print("=" * 64)

    print()
    print("New files:")

    for path in pending_files:
        print(
            f"  - {path.name}"
        )

    # ---------------------------------------------------------
    # 1. Bulk import
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("1. BULK IMPORT")
    print("=" * 64)

    imported_rows = run_bulk_import(
        bulk_dir=INBOX_DIR,
    )

    print(
        f"Imported/upserted rows: "
        f"{imported_rows:,}"
    )

    # ---------------------------------------------------------
    # 2. Training dataset
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("2. TRAINING DATASET")
    print("=" * 64)

    dataset_path = (
        make_training_dataset(
            history_hours=(
                24
                * TRAINING_HISTORY_DAYS
            ),
        )
    )

    print(
        f"Dataset: {dataset_path}"
    )

    # ---------------------------------------------------------
    # 3. Feature engineering
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("3. FEATURE ENGINEERING")
    print("=" * 64)

    features_path = (
        build_training_features(
            input_path=dataset_path,
        )
    )

    print(
        f"Features: {features_path}"
    )

    # ---------------------------------------------------------
    # 4. Multi-horizon training
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("4. MULTI-HORIZON TRAINING")
    print("=" * 64)

    training_results = (
        train_and_register_all_horizons(
            dataset_path=features_path,
        )
    )

    # ---------------------------------------------------------
    # 5. Quality gate
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("5. QUALITY GATE")
    print("=" * 64)

    promotion_results = (
        promote_all_candidates(
            dataset_path=features_path,
            metric="mae",
            min_baseline_improvement_pct=0.0,
            min_improvement_pct=0.5,
        )
    )

    # ---------------------------------------------------------
    # 6. Archive successfully processed files
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("6. ARCHIVE")
    print("=" * 64)

    archive_files(
        pending_files
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("REFRESH SUMMARY")
    print("=" * 64)

    for horizon in sorted(
        training_results
    ):
        training = (
            training_results[
                horizon
            ]
        )

        promotion = (
            promotion_results[
                horizon
            ]
        )

        print(
            f"+{horizon}h | "
            f"MAE={training['mae']:.4f} | "
            f"RMSE={training['rmse']:.4f} | "
            f"promoted="
            f"{promotion['promoted']} | "
            f"reason="
            f"{promotion['reason']}"
        )

    print()
    print(
        "Traffic model refresh "
        "completed successfully."
    )


if __name__ == "__main__":
    main()