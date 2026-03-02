"""Configuration management via pydantic-settings.

All configuration is loaded from environment variables and/or a .env file.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required ---
    airtable_api_key: str
    airtable_base_id: str
    firecrawl_api_key: str

    # --- Optional with defaults ---
    database_path: str = "data/companies.db"
    log_level: str = "INFO"
    max_retry_attempts: int = 2

    # --- LLM ---
    llm_validation_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str | None = None

    # --- Kagi ---
    kagi_api_key: str | None = None

    # --- LinkedIn / Playwright ---
    linkedin_headless: bool = False
    linkedin_profile_dir: str = "data/linkedin_profile"

    # ---- Validators ----

    @field_validator("airtable_api_key", "firecrawl_api_key")
    @classmethod
    def _non_empty_key(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("API key must not be empty")
        return v

    @field_validator("airtable_base_id")
    @classmethod
    def _valid_base_id(cls, v: str) -> str:
        if not re.match(r"^app[A-Za-z0-9]{4,}$", v):
            raise ValueError(
                "AIRTABLE_BASE_ID must start with 'app' followed by 4+ alphanumeric characters"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @field_validator("max_retry_attempts")
    @classmethod
    def _valid_retry_attempts(cls, v: int) -> int:
        if v < 0 or v > 5:
            raise ValueError("MAX_RETRY_ATTEMPTS must be between 0 and 5")
        return v

    @model_validator(mode="after")
    def _ensure_db_parent_dir(self) -> Config:
        """Auto-create parent directory for database file."""
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    # ---- Convenience properties ----

    @property
    def llm_enabled(self) -> bool:
        return self.llm_validation_enabled and bool(self.anthropic_api_key)

    @property
    def kagi_available(self) -> bool:
        return bool(self.kagi_api_key)
