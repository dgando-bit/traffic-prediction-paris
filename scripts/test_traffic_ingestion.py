import pandas as pd

from traffic_prediction.ingestion.traffic import ingest_traffic_data


output_path = ingest_traffic_data(
    output_path="data/raw/traffic/traffic_sample.parquet",
    order_by="t_1h desc",
    max_records=1000,
)

print(f"Fichier créé : {output_path}")

df = pd.read_parquet(output_path)

print()
print(df.info())

print()
print(df.head())

print()
print(f"Nombre de lignes : {len(df)}")

print()
print("Valeurs manquantes :")
print(df[["q", "k"]].isna().sum())