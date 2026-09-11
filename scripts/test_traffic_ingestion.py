import pandas as pd
from traffic_prediction.ingestion.traffic import (
    ingest_traffic_data,
)

OUTPUT_PATH = "data/raw/traffic/traffic_4836.parquet"


def main() -> None:
    output_path = ingest_traffic_data(
        output_path=OUTPUT_PATH,
        where='iu_ac = "4836"',
        order_by="t_1h asc",
        max_records=1000,
    )

    print(f"\nFichier créé : {output_path}")

    df = pd.read_parquet(output_path)

    print()
    print("=== Dataset ===")
    print(df.info())

    print()
    print("=== Aperçu ===")
    print(
        df[
            [
                "iu_ac",
                "t_1h",
                "timestamp_utc",
                "q",
                "k",
                "etat_trafic",
                "etat_barre",
            ]
        ].head(10)
    )

    print()
    print("=== Résumé ===")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de iu_ac : {df['iu_ac'].nunique()}")

    print()
    print("Valeurs manquantes :")
    print(
        df[
            [
                "q",
                "k",
                "timestamp_utc",
            ]
        ].isna().sum()
    )

    print()
    print("=== Période couverte ===")
    print("Début :", df["timestamp_utc"].min())
    print("Fin   :", df["timestamp_utc"].max())

    print()
    print("=== Contrôles ===")

    assert len(df) > 0
    assert "timestamp_utc" in df.columns
    assert df["iu_ac"].nunique() == 1
    assert df["iu_ac"].iloc[0] == "4836"
    assert df["timestamp_utc"].notna().all()

    print("✅ Traffic ingestion test OK")


if __name__ == "__main__":
    main()