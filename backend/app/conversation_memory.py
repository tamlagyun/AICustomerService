from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, *, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self._messages: dict[str, list[ConversationMessage]] = {}
        self._lock = RLock()

    def append_message(self, session_id: str, role: str, content: str) -> None:
        normalized_content = content.strip()
        if not session_id or not normalized_content:
            return

        with self._lock:
            messages = self._messages.setdefault(session_id, [])
            messages.append(ConversationMessage(role=role, content=normalized_content))
            if len(messages) > self.max_messages:
                self._messages[session_id] = messages[-self.max_messages :]

    def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        with self._lock:
            messages = list(self._messages.get(session_id, []))

        if limit is None:
            return messages
        return messages[-limit:]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._messages.pop(session_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._messages.clear()


_conversation_memory = ConversationMemory(max_messages=10)


def get_conversation_memory() -> ConversationMemory:
    return _conversation_memory
