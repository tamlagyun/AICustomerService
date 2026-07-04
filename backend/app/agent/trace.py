from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter


class TraceEventType(StrEnum):
    NODE_STARTED = "node.started"
    NODE_FINISHED = "node.finished"
    LLM_STARTED = "llm.started"
    LLM_FINISHED = "llm.finished"
    PLANNER_STARTED = "planner.started"
    PLANNER_FINISHED = "planner.finished"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    TOOL_FAILED = "tool.failed"
    ERROR = "error"


@dataclass(frozen=True)
class AgentTraceEvent:
    event_type: TraceEventType
    name: str
    timestamp: str
    duration_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    error: str = ""


class AgentTrace:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or perf_counter
        self._created_at = self._monotonic_clock()
        self.events: list[AgentTraceEvent] = []

    def record_started(
        self,
        event_type: TraceEventType,
        name: str,
        metadata: dict[str, object] | None = None,
    ) -> float:
        started_at = self._monotonic_clock()
        self.record_event(event_type, name, metadata=metadata)
        return started_at

    def record_finished(
        self,
        event_type: TraceEventType,
        name: str,
        started_at: float | None,
        metadata: dict[str, object] | None = None,
        error: str = "",
    ) -> None:
        duration_ms = None
        if started_at is not None:
            duration_ms = max(0.0, (self._monotonic_clock() - started_at) * 1000)
        self.record_event(
            event_type,
            name,
            duration_ms=duration_ms,
            metadata=metadata,
            error=error,
        )

    def record_error(
        self,
        name: str,
        error: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.record_event(
            TraceEventType.ERROR,
            name,
            metadata=metadata,
            error=error,
        )

    def record_event(
        self,
        event_type: TraceEventType,
        name: str,
        *,
        duration_ms: float | None = None,
        metadata: dict[str, object] | None = None,
        error: str = "",
    ) -> None:
        self.events.append(
            AgentTraceEvent(
                event_type=event_type,
                name=name,
                timestamp=self._clock().isoformat(),
                duration_ms=duration_ms,
                metadata=metadata or {},
                error=error,
            )
        )

    def summary(self) -> dict[str, object]:
        errors = [
            {"name": event.name, "error": event.error}
            for event in self.events
            if event.error
        ]
        return {
            "trace_event_count": len(self.events),
            "trace_errors": errors,
            "trace_duration_ms": max(0.0, (self._monotonic_clock() - self._created_at) * 1000),
        }
