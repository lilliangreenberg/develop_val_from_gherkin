"""Tests for configuration module."""

import os
import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no .env file interferes and clear relevant env vars."""
    # Point env_file to a nonexistent path so it doesn't load a real .env
    monkeypatch.chdir("/tmp")
    # Clear any relevant env vars that might leak in
    for var in [
        "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "FIRECRAWL_API_KEY",
        "DATABASE_PATH", "LOG_LEVEL", "MAX_RETRY_ATTEMPTS",
        "LLM_VALIDATION_ENABLED", "LLM_MODEL", "ANTHROPIC_API_KEY",
        "KAGI_API_KEY", "LINKEDIN_HEADLESS", "LINKEDIN_PROFILE_DIR",
    ]:
        monkeypatch.delenv(var, raising=False)


def _set_required_env(monkeypatch, overrides=None):
    """Set required env vars plus any overrides."""
    defaults = {
        "AIRTABLE_API_KEY": "pat.test_key_123",
        "AIRTABLE_BASE_ID": "appABC123DEF456",
        "FIRECRAWL_API_KEY": "fc-test_key_456",
    }
    if overrides:
        defaults.update(overrides)
    for k, v in defaults.items():
        monkeypatch.setenv(k, str(v))


def _load_config():
    from valuation_tool.config import Config
    return Config(_env_file=None)


class TestConfig:
    """Tests matching configuration.feature scenarios."""

    def test_all_required_variables_set(self, monkeypatch):
        _set_required_env(monkeypatch)
        config = _load_config()
        assert config.airtable_api_key == "pat.test_key_123"
        assert config.airtable_base_id == "appABC123DEF456"
        assert config.firecrawl_api_key == "fc-test_key_456"

    @pytest.mark.parametrize("missing_var", [
        "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "FIRECRAWL_API_KEY"
    ])
    def test_missing_required_variable(self, monkeypatch, missing_var):
        _set_required_env(monkeypatch)
        monkeypatch.delenv(missing_var)
        with pytest.raises(ValidationError):
            _load_config()

    def test_empty_api_key_rejected(self, monkeypatch):
        _set_required_env(monkeypatch, {"AIRTABLE_API_KEY": ""})
        with pytest.raises(ValidationError):
            _load_config()

    @pytest.mark.parametrize("base_id,should_pass", [
        ("appABC123DEF456", True),
        ("appAbCdEfGhIjKlMnO", True),
        ("app12345", True),
        ("tblABC123", False),
        ("ABC123DEF456", False),
        ("app", False),
        ("app!@#$%", False),
    ])
    def test_airtable_base_id_format(self, monkeypatch, base_id, should_pass):
        _set_required_env(monkeypatch, {"AIRTABLE_BASE_ID": base_id})
        if should_pass:
            config = _load_config()
            assert config.airtable_base_id == base_id
        else:
            with pytest.raises(ValidationError):
                _load_config()

    def test_default_values(self, monkeypatch):
        _set_required_env(monkeypatch)
        config = _load_config()
        assert config.database_path == "data/companies.db"
        assert config.log_level == "INFO"
        assert config.max_retry_attempts == 2
        assert config.llm_validation_enabled is False
        assert config.llm_model == "claude-haiku-4-5-20251001"
        assert config.linkedin_headless is False
        assert config.linkedin_profile_dir == "data/linkedin_profile"

    def test_custom_database_path(self, monkeypatch):
        _set_required_env(monkeypatch, {"DATABASE_PATH": "/custom/path/mydb.sqlite"})
        config = _load_config()
        assert config.database_path == "/custom/path/mydb.sqlite"

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_configure_log_level(self, monkeypatch, level):
        _set_required_env(monkeypatch, {"LOG_LEVEL": level})
        config = _load_config()
        assert config.log_level == level

    @pytest.mark.parametrize("value,should_pass", [
        (0, True), (1, True), (2, True), (5, True),
        (-1, False), (6, False), (100, False),
    ])
    def test_max_retry_attempts_range(self, monkeypatch, value, should_pass):
        _set_required_env(monkeypatch, {"MAX_RETRY_ATTEMPTS": str(value)})
        if should_pass:
            config = _load_config()
            assert config.max_retry_attempts == value
        else:
            with pytest.raises(ValidationError):
                _load_config()

    def test_llm_disabled_by_default(self, monkeypatch):
        _set_required_env(monkeypatch)
        config = _load_config()
        assert config.llm_validation_enabled is False
        assert config.llm_enabled is False

    def test_llm_enabled_with_key(self, monkeypatch):
        _set_required_env(monkeypatch, {
            "LLM_VALIDATION_ENABLED": "true",
            "ANTHROPIC_API_KEY": "sk-test-key-123",
            "LLM_MODEL": "claude-haiku-4-5-20251001",
        })
        config = _load_config()
        assert config.llm_enabled is True
        assert config.llm_model == "claude-haiku-4-5-20251001"

    def test_kagi_optional(self, monkeypatch):
        _set_required_env(monkeypatch)
        config = _load_config()
        assert config.kagi_available is False

    def test_kagi_configured(self, monkeypatch):
        _set_required_env(monkeypatch, {"KAGI_API_KEY": "test-kagi-key-123"})
        config = _load_config()
        assert config.kagi_available is True

    def test_linkedin_headless_configurable(self, monkeypatch):
        _set_required_env(monkeypatch, {"LINKEDIN_HEADLESS": "true"})
        config = _load_config()
        assert config.linkedin_headless is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("airtable_api_key", "test_key")
        monkeypatch.setenv("AIRTABLE_BASE_ID", "appABC123DEF456")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        config = _load_config()
        assert config.airtable_api_key == "test_key"

    def test_unknown_variables_ignored(self, monkeypatch):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("CUSTOM_VAR", "value")
        config = _load_config()
        assert not hasattr(config, "custom_var")
