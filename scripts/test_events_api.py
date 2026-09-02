import httpx


EVENTS_API_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/"
    "datasets/que-faire-a-paris-/records"
)


def test_period(
    start: str,
    end: str,
) -> None:
    where = (
        f'date_start <= "{end}" '
        f'and date_end >= "{start}"'
    )

    response = httpx.get(
        EVENTS_API_URL,
        params={
            "where": where,
            "limit": 1,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    payload = response.json()

    print(
        f"{start} -> {end}: "
        f"{payload.get('total_count', 0)} events"
    )


def main() -> None:
    periods = [
        (
            "2025-07-01T00:00:00Z",
            "2025-07-31T23:59:59Z",
        ),
        (
            "2025-10-01T00:00:00Z",
            "2025-10-31T23:59:59Z",
        ),
        (
            "2026-01-01T00:00:00Z",
            "2026-01-31T23:59:59Z",
        ),
        (
            "2026-04-01T00:00:00Z",
            "2026-04-30T23:59:59Z",
        ),
        (
            "2026-06-01T00:00:00Z",
            "2026-06-16T23:59:59Z",
        ),
    ]

    print("=== Event coverage ===")
    print()

    for start, end in periods:
        test_period(
            start,
            end,
        )


if __name__ == "__main__":
    main()