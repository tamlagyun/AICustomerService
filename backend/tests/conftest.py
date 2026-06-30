import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def disable_external_services(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MYSQL_ENABLED", "false")
    monkeypatch.setenv("LLM_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
