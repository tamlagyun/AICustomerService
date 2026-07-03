from collections.abc import Generator

import pytest

from app.config import get_settings
from app.prompt_registry import PromptNotFoundError, get_prompt_versions, load_prompt


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_load_prompt_reads_default_decision_prompt() -> None:
    prompt = load_prompt("decision", "v1.0")

    assert "游戏客服 Agent 的决策器" in prompt
    assert "mysql_players_list" in prompt


def test_get_prompt_versions_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPT_DECISION_VERSION", "v1.1")
    monkeypatch.setenv("PROMPT_FOLLOWUP_DECISION_VERSION", "v1.2")
    monkeypatch.setenv("PROMPT_FINAL_REPLY_VERSION", "v1.3")
    get_settings.cache_clear()

    versions = get_prompt_versions(get_settings())

    assert versions == {
        "decision": "v1.1",
        "followup_decision": "v1.2",
        "final_reply": "v1.3",
    }


def test_load_prompt_raises_clear_error_for_missing_version() -> None:
    with pytest.raises(PromptNotFoundError, match="decision.*missing"):
        load_prompt("decision", "missing")
