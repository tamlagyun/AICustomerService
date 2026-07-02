from app.config import Settings, get_settings
from app.llm import OpenAICompatibleLLMClient, build_llm_client


def test_build_llm_client_uses_requested_deepseek_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_ALLOWED_PROVIDERS", "deepseek,qwen")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    get_settings.cache_clear()

    client = build_llm_client("deepseek")

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_key == "deepseek-key"
    assert client.model == "deepseek-v4-flash"


def test_build_llm_client_keeps_legacy_deepseek_model_when_new_fields_are_unset(
    monkeypatch,
) -> None:
    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="deepseek",
        llm_default_provider="deepseek",
        llm_allowed_providers="deepseek,qwen",
        llm_base_url="https://legacy.deepseek.example/v1",
        llm_api_key="legacy-key",
        llm_model="legacy-deepseek-model",
    )
    monkeypatch.setattr("app.llm.get_settings", lambda: settings)

    client = build_llm_client("deepseek")

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.base_url == "https://legacy.deepseek.example/v1"
    assert client.api_key == "legacy-key"
    assert client.model == "legacy-deepseek-model"


def test_build_llm_client_uses_requested_qwen_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_ALLOWED_PROVIDERS", "deepseek,qwen")
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen-plus")
    get_settings.cache_clear()

    client = build_llm_client("qwen")

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert client.api_key == "qwen-key"
    assert client.model == "qwen-plus"


def test_build_llm_client_falls_back_to_default_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "qwen")
    monkeypatch.setenv("LLM_ALLOWED_PROVIDERS", "deepseek,qwen")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen-plus")
    get_settings.cache_clear()

    client = build_llm_client("unknown")

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_key == "qwen-key"
    assert client.model == "qwen-plus"
