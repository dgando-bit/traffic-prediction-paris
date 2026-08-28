from pathlib import Path

import pandas as pd

from traffic_prediction.ingestion.traffic import (
    fetch_traffic_data,
    normalize_traffic_data,
)


ROAD_IDS = [
    "4632",
    "4634",
    "1029",
    "1030",
    "1031",
    "1032",
    "1067",
    "1072",
    "4630",
    "4633",
    "4637",
    "985",
    "1033",
    "1034",
    "1111",
    "1112",
    "1222",
    "1043",
    "1044",
    "1038",
]

MAX_RECORDS_PER_ROAD = 8000

OUTPUT_PATH = Path(
    "data/raw/traffic/traffic_multi_20_long.parquet"
)


def main() -> None:
    frames: list[pd.DataFrame] = []

    for index, iu_ac in enumerate(
        ROAD_IDS,
        start=1,
    ):
        print(
            f"[{index:02}/{len(ROAD_IDS)}] "
            f"Téléchargement iu_ac={iu_ac}"
        )

        raw_records = fetch_traffic_data(
            where=f'iu_ac = "{iu_ac}"',
            order_by="t_1h asc",
            max_records=MAX_RECORDS_PER_ROAD,
        )

        df = normalize_traffic_data(
            raw_records
        )

        print(
            f"     {len(df)} lignes"
        )

        frames.append(df)

    dataset = pd.concat(
        frames,
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        [
            "timestamp_utc",
            "iu_ac",
        ]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=== Multi-road dataset ===")

    print(
        f"Nombre de lignes : {len(dataset)}"
    )

    print(
        f"Nombre de iu_ac : "
        f"{dataset['iu_ac'].nunique()}"
    )

    print(
        "Période :",
        dataset["timestamp_utc"].min(),
        "->",
        dataset["timestamp_utc"].max(),
    )

    print()
    print("Observations par tronçon :")

    print(
        dataset.groupby("iu_ac")
        .size()
        .sort_values(ascending=False)
    )

    print()
    print(
        f"Dataset sauvegardé : {OUTPUT_PATH}"
    )

    assert len(dataset) > 0

    assert (
        dataset["iu_ac"].nunique()
        == len(ROAD_IDS)
    )

    assert set(
        dataset["iu_ac"].unique()
    ) == set(ROAD_IDS)

    print()
    print(
        "✅ Multi-road ingestion test OK"
    )


if __name__ == "__main__":
    main()