from fastapi.testclient import TestClient

from app.main import app


def test_chat_returns_basic_agent_reply() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-1",
            "player_id": "player-1",
            "message": "我的角色卡住了",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["handoff"] is False
    assert "我的角色卡住了" in data["reply"]


def test_chat_marks_manual_handoff_for_appeal() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-1",
            "player_id": "player-1",
            "message": "我要申诉，转人工客服",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["handoff"] is True
    assert "转人工客服" in data["reply"]


def test_chat_returns_knowledge_source_for_recharge_question() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-1",
            "player_id": "player-1",
            "message": "充值不到账怎么办",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["handoff"] is False
    assert data["sources"][0]["source_type"] == "knowledge_base"
    assert data["sources"][0]["reference"] == "sample.md#充值未到账怎么办"
    assert "订单号" in data["reply"]


def test_chat_stream_returns_sse_events() -> None:
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "session_id": "session-1",
            "player_id": "player-1",
            "message": "我的角色卡住了",
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: status" in body
    assert "正在检查安全策略" in body
    assert "正在判断是否需要调用工具" in body
    assert "正在准备回复" in body
    assert "event: token" in body
    assert "event: done" in body
    assert "我的角色卡住了" in body
