import httpx

from shared.config import get_settings


N_ROADS = 20


def main() -> None:
    settings = get_settings()

    params = {
        "select": "iu_ac, count(*) as observations",
        "group_by": "iu_ac",
        "order_by": "observations desc",
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
        iu_ac = str(result["iu_ac"]).strip()

        if not iu_ac.isdigit():
            continue

        roads.append(
            {
                "iu_ac": iu_ac,
                "observations": result["observations"],
            }
        )

        if len(roads) == N_ROADS:
            break

    print(
        f"=== Top {len(roads)} road segments ==="
    )

    print()

    for road in roads:
        print(
            f"{road['iu_ac']:>8}  "
            f"{road['observations']:>8} observations"
        )

    print()
    print("ROAD_IDS = [")

    for road in roads:
        print(
            f'    "{road["iu_ac"]}",'
        )

    print("]")


if __name__ == "__main__":
    main()