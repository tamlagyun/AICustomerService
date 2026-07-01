import pytest

from app.config import get_settings
from app.conversation_memory import get_conversation_memory


@pytest.fixture(autouse=True)
def disable_external_services(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MYSQL_ENABLED", "false")
    monkeypatch.setenv("LLM_ENABLED", "false")
    get_conversation_memory().clear_all()
    get_settings.cache_clear()
    yield
    get_conversation_memory().clear_all()
    get_settings.cache_clear()
