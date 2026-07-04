from app.agent.customer_service import run_customer_service_agent
from app.agent.decision import AgentAction, AgentDecision
from app.rag.chroma_store import ChromaIndexNotReady
from tests.fakes import FakeLLMClient


async def test_llm_agent_uses_knowledge_action_and_summarizes_tool_result() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.KNOWLEDGE_BASE,
            reason="玩家询问充值未到账",
        ),
        final_reply="请提供订单号、充值时间、服务器和角色 ID。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        player_id="player-1",
        message="充值不到账怎么办",
        llm_client=llm_client,
    )

    assert response.reply == "请提供订单号、充值时间、服务器和角色 ID。"
    assert response.sources
    assert response.sources[0].source_type == "knowledge_base"
    assert llm_client.final_messages is not None
    assert "工具结果" in llm_client.final_messages[-1]["content"]


async def test_llm_clarification_for_recharge_issue_uses_knowledge_base() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.ASK_CLARIFICATION,
            reason="错误地要求玩家 ID",
            direct_reply="请问您的玩家ID或角色ID是什么？",
        ),
        final_reply="充值未到账时请先确认订单号、充值时间、服务器和角色 ID。",
    )

    response = await run_customer_service_agent(
        session_id="recharge-knowledge-override-session",
        message="充值不到账怎么办？",
        llm_client=llm_client,
    )

    assert response.sources
    assert response.sources[0].source_type == "knowledge_base"
    assert "充值未到账" in response.reply


async def test_llm_direct_refusal_uses_exact_knowledge_base_match() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="模型把知识库问题误判为应拒答闲聊",
            direct_reply="抱歉，我无法回答这个问题。",
        ),
        final_reply="不应该再次调用大模型生成",
    )

    response = await run_customer_service_agent(
        session_id="exact-knowledge-override-session",
        message="你吃过屎吗？",
        llm_client=llm_client,
    )

    assert response.sources
    assert response.sources[0].source_type == "knowledge_base"
    assert response.sources[0].reference == "sample.md#你吃过屎吗？"
    assert "什么味道" in response.reply
    assert llm_client.final_messages is None


async def test_llm_agent_uses_selected_vector_knowledge_source(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeKnowledgeBaseSearch:
        def __init__(self, root_dir, *, knowledge_source=None, **kwargs) -> None:
            captured["root_dir"] = str(root_dir)
            captured["knowledge_source"] = knowledge_source
            captured["kwargs"] = kwargs

        def search(self, query: str, limit: int = 3):
            captured["query"] = query
            captured["limit"] = limit
            from app.knowledge_base import KnowledgeChunk

            return [
                KnowledgeChunk(
                    title="充值未到账怎么办",
                    content="请提供订单号、充值时间、服务器和角色 ID。",
                    reference="sample.md#充值未到账怎么办",
                )
            ]

    monkeypatch.setattr("app.agent.customer_service.KnowledgeBaseSearch", FakeKnowledgeBaseSearch)
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.KNOWLEDGE_BASE,
            reason="玩家询问充值未到账",
        ),
        final_reply="向量库答案：请提供订单号、充值时间、服务器和角色 ID。",
    )

    response = await run_customer_service_agent(
        session_id="vector-source-session",
        message="我充钱了但是没到账",
        knowledge_source="vector",
        llm_client=llm_client,
    )

    assert captured["knowledge_source"] == "vector"
    assert captured["query"] == "我充钱了但是没到账"
    assert captured["limit"] == 1
    assert response.sources[0].source_type == "knowledge_base"
    assert response.reply == "向量库答案：请提供订单号、充值时间、服务器和角色 ID。"


async def test_llm_agent_returns_clear_message_when_vector_index_is_missing(monkeypatch) -> None:
    class MissingVectorKnowledgeBaseSearch:
        def __init__(self, root_dir, *, knowledge_source=None, **kwargs) -> None:
            pass

        def search(self, query: str, limit: int = 3):
            raise ChromaIndexNotReady("向量知识库尚未建立")

    monkeypatch.setattr(
        "app.agent.customer_service.KnowledgeBaseSearch",
        MissingVectorKnowledgeBaseSearch,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.KNOWLEDGE_BASE,
            reason="玩家询问知识库问题",
        ),
        final_reply="不应该调用最终生成",
    )

    response = await run_customer_service_agent(
        session_id="missing-vector-session",
        message="充值不到账怎么办？",
        knowledge_source="vector",
        llm_client=llm_client,
    )

    assert "向量知识库尚未建立" in response.reply
    assert "Agent 评测" in response.reply
    assert response.sources == []
    assert llm_client.final_messages is None
