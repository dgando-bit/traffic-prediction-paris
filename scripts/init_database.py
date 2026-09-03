from traffic_prediction.storage.database import ENGINE
from traffic_prediction.storage.models import Base


def main() -> None:
    Base.metadata.create_all(
        bind=ENGINE
    )

    print(
        "Database tables created."
    )


if __name__ == "__main__":
    main()