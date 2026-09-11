from pathlib import Path

import pandas as pd
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    upsert_traffic_dataframe,
)

INPUT_PATH = Path(
    "data/interim/traffic_multi_20_long_clean.parquet"
)


def main() -> None:
    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Loaded {len(df)} traffic rows"
    )

    with get_db_session() as session:
        inserted = upsert_traffic_dataframe(
            session,
            df,
        )

    print(
        f"Upserted {inserted} rows "
        "into PostgreSQL"
    )


if __name__ == "__main__":
    main()