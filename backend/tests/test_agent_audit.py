import json

from app.agent_audit import write_agent_audit_event
from app.config import Settings


def test_write_agent_audit_event_appends_json_line(tmp_path) -> None:
    settings = Settings(log_dir=str(tmp_path), agent_audit_log_enabled=True)

    audit_file = write_agent_audit_event(
        settings,
        {
            "session_id": "session-1",
            "player_id": "1",
            "message": "查询我的资料",
            "reply": "玩家资料已查询。",
            "tools": [{"tool": "mysql_player_profile", "status": "found"}],
        },
    )

    lines = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["timestamp"]
    assert payload["session_id"] == "session-1"
    assert payload["tools"] == [{"tool": "mysql_player_profile", "status": "found"}]


def test_write_agent_audit_event_returns_path_without_writing_when_disabled(tmp_path) -> None:
    settings = Settings(log_dir=str(tmp_path), agent_audit_log_enabled=False)

    audit_file = write_agent_audit_event(settings, {"session_id": "session-1"})

    assert audit_file == tmp_path / "agent_audit.jsonl"
    assert not audit_file.exists()

