from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings shared across the project."""

    project_name: str = "Traffic Prediction Paris"

    paris_open_data_base_url: str = "https://opendata.paris.fr"
    traffic_dataset: str = "comptages-routiers-permanents"

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")

    http_timeout: float = 30.0
    api_page_size: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TP_",
        extra="ignore",
    )

    @property
    def traffic_api_url(self) -> str:
        return (
            f"{self.paris_open_data_base_url}/api/explore/v2.1/catalog/"
            f"datasets/{self.traffic_dataset}/records"
        )

    @property
    def raw_traffic_dir(self) -> Path:
        return self.raw_data_dir / "traffic"


@lru_cache
def get_settings() -> Settings:
    return Settings()