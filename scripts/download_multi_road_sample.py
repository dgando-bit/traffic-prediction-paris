from pathlib import Path

import pandas as pd

from traffic_prediction.ingestion.traffic import (
    fetch_traffic_data,
    normalize_traffic_data,
)


ROAD_IDS = [
    "1586",
    "4225",
    "4596",
    "5758",
    "782",
    "1",
    "10",
    "100",
    "1001",
    "1006",
    "1007",
    "101",
    "1010",
    "1011",
    "1017",
    "1018",
    "1019",
    "1020",
    "1023",
    "1024",
]

MAX_RECORDS_PER_ROAD = 1000

OUTPUT_PATH = Path(
    "data/raw/traffic/traffic_multi_20.parquet"
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