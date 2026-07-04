import json

from fastapi.testclient import TestClient

from app.agent.checkpoint import (
    AgentCheckpoint,
    CheckpointStore,
    write_chat_checkpoint_event,
)
from app.agent.decision import AgentAction, AgentDecision
from app.agent.trace import AgentTrace
from app.config import Settings, get_settings
from app.main import app
from app.schemas import ChatResponse
from app.agent.customer_service import run_customer_service_agent


def test_checkpoint_store_does_not_write_when_disabled(tmp_path) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        agent_checkpoint_enabled=False,
        agent_checkpoint_file="checkpoints.jsonl",
    )
    store = CheckpointStore(settings)

    checkpoint_file = store.append(
        AgentCheckpoint(session_id="session-1", player_id=None, message="你好")
    )

    assert checkpoint_file == tmp_path / "checkpoints.jsonl"
    assert not checkpoint_file.exists()


def test_checkpoint_store_writes_jsonl_when_enabled(tmp_path) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        agent_checkpoint_enabled=True,
        agent_checkpoint_file="checkpoints.jsonl",
    )
    store = CheckpointStore(settings)

    store.append(
        AgentCheckpoint(
            session_id="session-1",
            player_id="1",
            message="查询我的资料",
            payload={"llm_action": "mysql_player_profile"},
        )
    )

    checkpoint_file = tmp_path / "checkpoints.jsonl"
    payload = json.loads(checkpoint_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["timestamp"]
    assert payload["session_id"] == "session-1"
    assert payload["player_id"] == "1"
    assert payload["message"] == "查询我的资料"
    assert payload["llm_action"] == "mysql_player_profile"


def test_checkpoint_store_lists_recent_checkpoints_by_session(tmp_path) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        agent_checkpoint_enabled=True,
        agent_checkpoint_file="checkpoints.jsonl",
    )
    store = CheckpointStore(settings)
    store.append(AgentCheckpoint(session_id="session-a", player_id=None, message="a1"))
    store.append(AgentCheckpoint(session_id="session-b", player_id=None, message="b1"))
    store.append(AgentCheckpoint(session_id="session-a", player_id=None, message="a2"))

    checkpoints = store.list_recent(session_id="session-a", limit=10)

    assert [checkpoint["message"] for checkpoint in checkpoints] == ["a1", "a2"]


def test_write_chat_checkpoint_event_captures_agent_state(tmp_path) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        agent_checkpoint_enabled=True,
        agent_checkpoint_file="checkpoints.jsonl",
    )
    trace = AgentTrace()
    trace.record_error("retrieve_player_data", "timeout")

    write_chat_checkpoint_event(
        settings,
        session_id="session-1",
        player_id="1",
        message="查询我的资料",
        response=ChatResponse(reply="已查询"),
        final_state={
            "llm_decision": AgentDecision(
                action=AgentAction.MYSQL_PLAYER_PROFILE,
                reason="query",
                arguments={"player_id": "1"},
            ),
            "agent_trace": trace,
            "completed_plan_steps": [{"action": "mysql_player_profile"}],
        },
    )

    checkpoint_file = tmp_path / "checkpoints.jsonl"
    payload = json.loads(checkpoint_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["session_id"] == "session-1"
    assert payload["reply"] == "已查询"
    assert payload["llm_action"] == "mysql_player_profile"
    assert payload["completed_plan_steps"] == [{"action": "mysql_player_profile"}]
    assert payload["trace_errors"] == [
        {"name": "retrieve_player_data", "error": "timeout"}
    ]


def test_checkpoints_api_requires_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(app)

    response = client.get("/api/checkpoints?session_id=session-1")

    assert response.status_code == 403


def test_checkpoints_api_lists_session_checkpoints_when_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_ENABLED", "true")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_CHECKPOINT_FILE", "checkpoints.jsonl")
    get_settings.cache_clear()
    store = CheckpointStore(get_settings())
    store.append(AgentCheckpoint(session_id="session-1", player_id=None, message="a"))
    store.append(AgentCheckpoint(session_id="session-2", player_id=None, message="b"))
    client = TestClient(app)

    response = client.get("/api/checkpoints?session_id=session-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoints"][0]["session_id"] == "session-1"
    assert payload["checkpoints"][0]["message"] == "a"


async def test_customer_service_writes_checkpoint_when_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_ENABLED", "true")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_CHECKPOINT_FILE", "checkpoints.jsonl")
    get_settings.cache_clear()

    try:
        response = await run_customer_service_agent(
            session_id="checkpoint-session",
            player_id="1",
            message="我的角色卡住了",
            llm_client=None,
        )
    finally:
        get_settings.cache_clear()

    checkpoint_file = tmp_path / "checkpoints.jsonl"
    payload = json.loads(checkpoint_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["session_id"] == "checkpoint-session"
    assert payload["player_id"] == "1"
    assert payload["message"] == "我的角色卡住了"
    assert payload["reply"] == response.reply


async def test_customer_service_ignores_checkpoint_write_failures(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_ENABLED", "true")
    get_settings.cache_clear()

    def failing_write_checkpoint(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "app.agent.customer_service.write_chat_checkpoint_event",
        failing_write_checkpoint,
    )
    try:
        response = await run_customer_service_agent(
            session_id="checkpoint-failure-session",
            player_id=None,
            message="我的角色卡住了",
            llm_client=None,
        )
    finally:
        get_settings.cache_clear()

    assert "我的角色卡住了" in response.reply
