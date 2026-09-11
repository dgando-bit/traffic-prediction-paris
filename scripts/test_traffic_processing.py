import pandas as pd
from traffic_prediction.processing.cleaning import (
    process_traffic_file,
)

input_path = "data/raw/traffic/traffic_sample.parquet"
output_path = "data/interim/traffic_clean.parquet"

result = process_traffic_file(
    input_path=input_path,
    output_path=output_path,
)

print(f"\nFichier créé : {result}")

df = pd.read_parquet(result)

print()
print(df.info())

print()
print(
    df[
        [
            "iu_ac",
            "timestamp_utc",
            "timestamp_paris",
            "q",
            "k",
            "etat_trafic",
            "etat_barre",
            "is_valid_iu_ac",
            "has_q",
            "has_k",
            "has_complete_measurement",
        ]
    ].head(10)
)

print()
print("=== Data quality ===")

print(f"Nombre de lignes : {len(df)}")

print(
    "iu_ac invalides :",
    (~df["is_valid_iu_ac"]).sum(),
)

print(
    "q manquants :",
    (~df["has_q"]).sum(),
)

print(
    "k manquants :",
    (~df["has_k"]).sum(),
)

print(
    "Mesures q+k complètes :",
    df["has_complete_measurement"].sum(),
)