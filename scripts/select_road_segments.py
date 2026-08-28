import httpx

from shared.config import get_settings


N_ROADS = 20
MIN_K_COVERAGE = 0.90


def main() -> None:
    settings = get_settings()

    params = {
        "select": (
            "iu_ac, "
            "count(*) as observations, "
            "count(k) as k_available"
        ),
        "group_by": "iu_ac",
        "order_by": "k_available desc",
        "limit": 100,
    }

    response = httpx.get(
        settings.traffic_api_url,
        params=params,
        timeout=settings.http_timeout,
    )

    response.raise_for_status()

    payload = response.json()

    roads = []

    for result in payload["results"]:
        iu_ac = str(
            result["iu_ac"]
        ).strip()

        if not iu_ac.isdigit():
            continue

        observations = int(
            result["observations"]
        )

        k_available = int(
            result["k_available"]
        )

        if observations == 0:
            continue

        k_coverage = (
            k_available / observations
        )

        if k_coverage < MIN_K_COVERAGE:
            continue

        roads.append(
            {
                "iu_ac": iu_ac,
                "observations": observations,
                "k_available": k_available,
                "k_coverage": k_coverage,
            }
        )

        if len(roads) == N_ROADS:
            break

    print(
        f"=== Top {len(roads)} road segments ==="
    )

    print()

    print(
        f"{'iu_ac':>8}"
        f"{'rows':>10}"
        f"{'k':>10}"
        f"{'coverage':>12}"
    )

    print("-" * 40)

    for road in roads:
        print(
            f"{road['iu_ac']:>8}"
            f"{road['observations']:>10}"
            f"{road['k_available']:>10}"
            f"{road['k_coverage']:>11.1%}"
        )

    print()
    print("ROAD_IDS = [")

    for road in roads:
        print(
            f'    "{road["iu_ac"]}",'
        )

    print("]")

    assert len(roads) == N_ROADS

    assert all(
        road["k_coverage"]
        >= MIN_K_COVERAGE
        for road in roads
    )

    print()
    print(
        "✅ Road segment selection test OK"
    )


if __name__ == "__main__":
    main()