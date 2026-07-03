from datetime import UTC, datetime
import json
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import Settings
from app.logging_config import resolve_log_dir

_audit_lock = RLock()


def write_agent_audit_event(settings: Settings, event: dict[str, Any]) -> Path:
    audit_file = agent_audit_log_path(settings)
    if not settings.agent_audit_log_enabled:
        return audit_file

    audit_file.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(UTC).isoformat())
    with _audit_lock, audit_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str))
        file.write("\n")
    return audit_file


def agent_audit_log_path(settings: Settings) -> Path:
    return resolve_log_dir(settings.log_dir) / "agent_audit.jsonl"
