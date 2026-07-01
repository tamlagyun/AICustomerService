from app.conversation_memory import ConversationMemory


def test_conversation_memory_keeps_recent_messages_by_session() -> None:
    memory = ConversationMemory(max_messages=3)

    memory.append_message("session-1", "user", "第一句")
    memory.append_message("session-1", "assistant", "第二句")
    memory.append_message("session-1", "user", "第三句")
    memory.append_message("session-1", "assistant", "第四句")
    memory.append_message("session-2", "user", "另一个会话")

    session_1_messages = memory.get_recent_messages("session-1")
    session_2_messages = memory.get_recent_messages("session-2")

    assert [message.content for message in session_1_messages] == ["第二句", "第三句", "第四句"]
    assert [message.content for message in session_2_messages] == ["另一个会话"]


def test_conversation_memory_clear_session() -> None:
    memory = ConversationMemory(max_messages=10)
    memory.append_message("session-1", "user", "需要清理")

    memory.clear_session("session-1")

    assert memory.get_recent_messages("session-1") == []
