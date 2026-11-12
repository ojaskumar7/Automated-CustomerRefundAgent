"""Application configuration via pydantic-settings."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM advisor (never approves/executes refunds)
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"

    # Refund business rules
    refund_auto_approve_threshold: float = 50.0
    max_refunds_30d: int = 3
    return_window_days: int = 30

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    policies_dir: Path = data_dir / "policies"
    db_path: Path = data_dir / "orders.db"
    chroma_dir: Path = data_dir / "chroma"


settings = Settings()
