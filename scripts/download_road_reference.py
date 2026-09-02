from pathlib import Path

from traffic_prediction.ingestion.road_reference import (
    fetch_road_reference,
    normalize_road_reference,
    save_road_reference,
)


ROAD_IDS = [
    "4632", "4634", "1029", "1030", "1031",
    "1032", "1067", "1072", "4630", "4633",
    "4637", "985", "1033", "1034", "1111",
    "1112", "1222", "1043", "1044", "1038",
]

OUTPUT_PATH = Path(
    "data/raw/reference/road_reference.parquet"
)


def main() -> None:
    raw_df = fetch_road_reference(ROAD_IDS)

    df = normalize_road_reference(raw_df)

    print("=== Road reference ===")
    print(f"Rows         : {len(df)}")
    print(f"Unique roads : {df['iu_ac'].nunique()}")

    print(
        df[
            [
                "iu_ac",
                "libelle",
                "latitude",
                "longitude",
                "road_length_m",
                "iu_nd_amont",
                "iu_nd_aval",
            ]
        ]
        .sort_values("iu_ac")
        .to_string(index=False)
    )

    print("\n=== Missing values ===")
    print(
        df[
            [
                "latitude",
                "longitude",
                "road_length_m",
            ]
        ].isna().sum()
    )

    print("\n=== Road length statistics ===")
    print(df["road_length_m"].describe())

    save_road_reference(
        df,
        OUTPUT_PATH,
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()