from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ChatResponse


def test_evaluation_run_requires_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_EVAL_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    response = client.post("/api/evaluations/run", json={"model_provider": "deepseek"})

    assert response.status_code == 403


def test_evaluation_cases_returns_builtin_cases_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_EVAL_ENABLED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    response = client.get("/api/evaluations/cases")

    assert response.status_code == 200
    case_ids = {case["case_id"] for case in response.json()["cases"]}
    assert {"safety_refuse", "knowledge_recharge", "mysql_player_profile"} <= case_ids
    assert "knowledge_rag_semantic_recharge" in case_ids


def test_evaluation_run_skips_mysql_case_when_mysql_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_EVAL_ENABLED", "true")
    monkeypatch.setenv("MYSQL_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/api/evaluations/run",
        json={"model_provider": "deepseek", "case_ids": ["mysql_player_profile"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"total": 1, "passed": 0, "failed": 0, "skipped": 1}
    assert payload["results"][0]["status"] == "skipped"
    assert "mysql" in payload["results"][0]["error"].lower()


def test_evaluation_run_passes_use_planner_to_agent(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_EVAL_ENABLED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    captured: dict[str, object] = {}

    async def fake_run_customer_service_agent(**kwargs) -> ChatResponse:
        captured.update(kwargs)
        return ChatResponse(reply="不能提供 API key。", handoff=False)

    monkeypatch.setattr("app.evaluations.run_customer_service_agent", fake_run_customer_service_agent)
    client = TestClient(app)

    response = client.post(
        "/api/evaluations/run",
        json={"model_provider": "qwen", "use_planner": True, "case_ids": ["safety_refuse"]},
    )

    assert response.status_code == 200
    assert captured["model_provider"] == "qwen"
    assert captured["use_planner"] is True
    assert response.json()["results"][0]["status"] == "passed"
