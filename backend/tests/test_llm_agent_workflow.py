import asyncio
import json
import logging

import pytest

from app.agent.customer_service import run_customer_service_agent, stream_customer_service_agent
from app.agent.decision import AgentAction, AgentDecision
from app.config import get_settings
from app.prompt_registry import PromptNotFoundError
from tests.fakes import (
    FailingDecisionLLMClient,
    FailingFinalReplyLLMClient,
    FailingStreamLLMClient,
    FakeGuangzhouAttractionMapTools,
    FakeLLMClient,
    FakePlayerDataTools,
    MultiStepFakeLLMClient,
    PlannerFakeLLMClient,
)


async def test_llm_agent_calls_players_then_map_for_guangzhou_recommendations(
    monkeypatch,
) -> None:
    player_tools = FakePlayerDataTools()
    map_tools = FakeGuangzhouAttractionMapTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = MultiStepFakeLLMClient(
        decisions=[
            AgentDecision(
                action=AgentAction.MYSQL_PLAYERS_LIST,
                reason="先查询数据库中的玩家资料",
                arguments={"limit": 100},
                final_task="根据玩家特色和 desc 类型推荐广州景点，并说明游玩后的心情收获",
            ),
            AgentDecision(
                action=AgentAction.AMAP_PLACE_SEARCH,
                reason="玩家资料已获得，还需要查询广州旅游景点",
                arguments={
                    "keywords": "旅游景点",
                    "city": "广州",
                    "presentation": {"mode": "table"},
                },
                final_task="结合玩家资料和广州景点结果生成推荐",
            ),
        ],
        final_reply="已结合玩家特色和广州景点完成推荐。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="查询数据库中的玩家资料，并根据玩家特色和desc类型分析他们适合去广州什么景点游玩？游玩后大概收获什么心情？",
        llm_client=llm_client,
    )

    assert player_tools.requested_limit == 100
    assert map_tools.place_query == {"keywords": "旅游景点", "city": "广州", "types": None}
    assert len(llm_client.decision_messages) == 2
    assert "进攻型玩家" in llm_client.decision_messages[1][-1]["content"]
    assert response.reply == "已结合玩家特色和广州景点完成推荐。"
    assert response.sources[0].reference == "maps_text_search"
    recommendation_table = next(table for table in response.tables if table.title == "玩家景点推荐")
    assert recommendation_table.rows[0]["recommended_place"] == "广州塔"
    assert recommendation_table.rows[1]["recommended_place"] == "广东省博物馆"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "mysql_players_list" in final_prompt
    assert "maps_text_search" in final_prompt
    assert "广州塔" in final_prompt
    assert "进攻型玩家" in final_prompt


async def test_llm_clarification_for_players_attractions_uses_multi_step_tools(
    monkeypatch,
) -> None:
    player_tools = FakePlayerDataTools()
    map_tools = FakeGuangzhouAttractionMapTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.ASK_CLARIFICATION,
            reason="错误地要求单个玩家 ID",
            direct_reply="请问您的玩家ID是什么？",
        ),
        final_reply="已结合玩家 desc 和广州景点完成推荐。",
    )

    response = await run_customer_service_agent(
        session_id="llm-clarification-players-attractions-session",
        message="查询玩家资料并根据desc推荐广州景点，用表格显示",
        llm_client=llm_client,
    )

    assert player_tools.requested_limit == 100
    assert map_tools.place_query == {"keywords": "旅游景点", "city": "广州", "types": None}
    assert response.reply == "已结合玩家 desc 和广州景点完成推荐。"
    assert any(table.title == "玩家景点推荐" for table in response.tables)


async def test_planner_executes_player_then_map_steps_when_enabled(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    map_tools = FakeGuangzhouAttractionMapTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = PlannerFakeLLMClient(
        plan_reply=json.dumps(
            {
                "final_task": "combine players and attractions",
                "steps": [
                    {
                        "action": "mysql_players_list",
                        "reason": "need players",
                        "arguments": {"limit": 100},
                    },
                    {
                        "action": "amap_place_search",
                        "reason": "need Guangzhou attractions",
                        "arguments": {"keywords": "旅游景点", "city": "广州"},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        final_reply="Planner 已完成玩家和广州景点联合推荐。",
    )

    response = await run_customer_service_agent(
        session_id="planner-session",
        message="查询玩家资料并根据desc推荐广州景点",
        llm_client=llm_client,
        use_planner=True,
    )

    assert player_tools.requested_limit == 100
    assert map_tools.place_query == {"keywords": "旅游景点", "city": "广州", "types": None}
    assert llm_client.decision_messages == []
    assert len(llm_client.generate_messages) == 2
    assert response.reply == "Planner 已完成玩家和广州景点联合推荐。"
    assert response.sources[0].reference == "maps_text_search"


async def test_planner_clarification_for_players_attractions_uses_multi_step_tools(
    monkeypatch,
) -> None:
    player_tools = FakePlayerDataTools()
    map_tools = FakeGuangzhouAttractionMapTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = PlannerFakeLLMClient(
        plan_reply=json.dumps(
            {
                "steps": [
                    {
                        "action": "ask_clarification",
                        "reason": "错误地要求单个玩家 ID",
                        "direct_reply": "请问您的玩家ID是什么？",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        final_reply="已结合玩家 desc 和广州景点完成推荐。",
    )

    response = await run_customer_service_agent(
        session_id="planner-clarification-override-session",
        message="查询玩家资料并根据desc推荐广州景点，用表格显示",
        llm_client=llm_client,
        use_planner=True,
    )

    assert player_tools.requested_limit == 100
    assert map_tools.place_query == {"keywords": "旅游景点", "city": "广州", "types": None}
    assert response.reply == "已结合玩家 desc 和广州景点完成推荐。"
    assert any(table.title == "玩家景点推荐" for table in response.tables)


async def test_planner_is_not_used_when_request_flag_is_false(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    llm_client = PlannerFakeLLMClient(
        plan_reply=json.dumps(
            {"steps": [{"action": "mysql_players_list", "reason": "would be planner"}]},
            ensure_ascii=False,
        ),
        final_reply="不应该使用 Planner 最终回复",
    )

    response = await run_customer_service_agent(
        session_id="planner-disabled-session",
        message="查询数据库中所有的资料并且根据desc进行分类，用表格显示出来",
        llm_client=llm_client,
        use_planner=False,
    )

    assert len(llm_client.generate_messages) == 0
    assert len(llm_client.decision_messages) == 1
    assert player_tools.requested_limit == 100
    assert response.tables[0].title == "玩家列表"


@pytest.mark.parametrize(
    "plan_reply",
    [
        "not json",
        '{"steps":[{"action":"write_sql","reason":"bad"}]}',
        '{"steps":[]}',
    ],
)
async def test_planner_parse_error_falls_back_to_legacy_decision(monkeypatch, plan_reply: str) -> None:
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    llm_client = PlannerFakeLLMClient(
        plan_reply=plan_reply,
        final_reply="旧流程完成玩家列表查询。",
        fallback_decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYERS_LIST,
            reason="Planner 失败后继续使用旧流程",
        ),
    )

    response = await run_customer_service_agent(
        session_id="planner-invalid-json-session",
        message="查询数据库中所有的资料并且根据desc进行分类，用表格显示出来",
        llm_client=llm_client,
        use_planner=True,
    )

    assert len(llm_client.generate_messages) == 2
    assert len(llm_client.decision_messages) == 1
    assert player_tools.requested_limit == 100
    assert response.reply == "旧流程完成玩家列表查询。"
    assert response.tables[0].title == "玩家列表"


async def test_agent_emits_status_when_llm_decision_fails_then_uses_local_players_list(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(logging.ERROR, logger="app.agent.customer_service")
    caplog.set_level(logging.ERROR, logger="app.llm_middleware")
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    response = await run_customer_service_agent(
        session_id="session-1",
        message="查询数据库中所有的资料并且根据desc进行分类，用表格显示出来",
        llm_client=FailingDecisionLLMClient(),
        status_queue=status_queue,
    )

    statuses = []
    while not status_queue.empty():
        statuses.append(status_queue.get_nowait())

    assert any("大模型决策失败" in status for status in statuses)
    assert (
        "LLM call failed; operation=decide_action provider=injected model=unknown "
        "session_id=session-1"
    ) in caplog.text
    assert "LLM decision failed" in caplog.text
    assert "TimeoutError" in caplog.text
    assert player_tools.requested_limit == 100
    assert len(response.tables) == 1


async def test_agent_logs_when_llm_final_reply_fails(monkeypatch, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="app.agent.customer_service")
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="查询数据库中所有的资料并且根据desc进行分类，用表格显示出来",
        llm_client=FailingFinalReplyLLMClient(),
    )

    assert "LLM final reply failed" in caplog.text
    assert "TimeoutError" in caplog.text
    assert response.reply == "共查询到 2 条玩家数据，当前返回上限 100 条。"
    assert len(response.tables) == 1


async def test_stream_agent_logs_when_llm_streaming_reply_fails(monkeypatch, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="app.agent.customer_service")
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )

    events = [
        event
        async for event in stream_customer_service_agent(
            session_id="session-1",
            message="查询数据库中所有的资料并且根据desc进行分类，用表格显示出来",
            llm_client=FailingStreamLLMClient(),
        )
    ]

    assert "LLM streaming reply failed" in caplog.text
    assert "TimeoutError" in caplog.text
    assert any(
        event["event"] == "token"
        and event["data"] == {"text": "流式失败后使用普通生成回复"}
        for event in events
    )


async def test_llm_agent_direct_answer_does_not_call_final_generation() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="普通问候",
            direct_reply="你好，请描述你的问题。",
        ),
        final_reply="不应该使用这个回复",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        player_id="player-1",
        message="你好",
        llm_client=llm_client,
    )

    assert response.reply == "你好，请描述你的问题。"
    assert llm_client.final_messages is None


async def test_agent_builds_llm_client_with_requested_provider(monkeypatch) -> None:
    requested_providers: list[str | None] = []

    def build_client(model_provider: str | None = None):
        requested_providers.append(model_provider)
        return None

    monkeypatch.setattr("app.agent.customer_service.build_llm_client", build_client)

    await run_customer_service_agent(
        session_id="provider-session",
        message="你好",
        model_provider="qwen",
    )

    assert requested_providers == ["qwen"]


async def test_llm_agent_logs_prompt_versions(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.agent.customer_service")
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="普通问候",
            direct_reply="你好，请描述你的问题。",
        ),
        final_reply="不应该使用这个回复",
    )

    await run_customer_service_agent(
        session_id="prompt-version-session",
        message="你好",
        llm_client=llm_client,
    )

    assert (
        "Prompt versions selected; session_id=prompt-version-session "
        "decision=v1.0 planner=v1.0 followup_decision=v1.0 final_reply=v1.0"
    ) in caplog.text


async def test_customer_service_writes_agent_audit_log(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_AUDIT_LOG_ENABLED", "true")
    get_settings.cache_clear()
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYER_PROFILE,
            reason="玩家要求查询资料",
            arguments={"player_id": "1"},
            final_task="总结玩家资料",
        ),
        final_reply="玩家 ai大名 喜欢研究机制，适合策略型玩法。",
    )

    try:
        response = await run_customer_service_agent(
            session_id="audit-session",
            player_id="1",
            message="player_id=1请查询我的资料",
            llm_client=llm_client,
        )
    finally:
        get_settings.cache_clear()

    audit_file = tmp_path / "agent_audit.jsonl"
    payload = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["event_type"] == "chat_completed"
    assert payload["session_id"] == "audit-session"
    assert payload["player_id"] == "1"
    assert payload["message"] == "player_id=1请查询我的资料"
    assert payload["reply"] == response.reply
    assert payload["handoff"] is False
    assert payload["llm_action"] == "mysql_player_profile"
    assert payload["tools"][0]["tool"] == "mysql_player_profile"
    assert payload["tools"][0]["status"] == "found"
    assert payload["sources"] == []


async def test_llm_agent_fails_clearly_when_configured_prompt_version_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PROMPT_DECISION_VERSION", "missing")
    get_settings.cache_clear()
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="普通问候",
            direct_reply="你好，请描述你的问题。",
        ),
        final_reply="不应该使用这个回复",
    )

    try:
        with pytest.raises(PromptNotFoundError, match="decision.*missing"):
            await run_customer_service_agent(
                session_id="prompt-missing-session",
                message="你好",
                llm_client=llm_client,
            )
    finally:
        get_settings.cache_clear()


async def test_llm_agent_decision_prompt_includes_current_streaming_capability() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="回答当前系统能力",
            direct_reply="当前前端到后端聊天已采用 SSE 流式输出。",
        ),
        final_reply="不应该使用这个回复",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="你好",
        llm_client=llm_client,
    )

    assert response.reply == "当前前端到后端聊天已采用 SSE 流式输出。"
    assert llm_client.decision_messages is not None
    decision_prompt = llm_client.decision_messages[0]["content"]
    assert "SSE 流式输出" in decision_prompt
    assert "/api/chat/stream" in decision_prompt


async def test_agent_answers_streaming_capability_from_system_fact() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="错误回答",
            direct_reply="还未实现流式输出。",
        ),
        final_reply="不应该使用这个回复",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="你们现在是否采用流式输出？",
        llm_client=llm_client,
    )

    assert "已采用 SSE 流式输出" in response.reply
    assert "/api/chat/stream" in response.reply


async def test_llm_agent_includes_same_session_history_in_decision_prompt() -> None:
    first_turn_llm = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="记录玩家 ID",
            direct_reply="我记住了，你的 ID 是 1。",
        ),
        final_reply="不应该使用这个回复",
    )
    await run_customer_service_agent(
        session_id="memory-session",
        message="我的 ID 是 1",
        llm_client=first_turn_llm,
    )
    second_turn_llm = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYER_PROFILE,
            reason="根据历史中的玩家 ID 查询资料",
            arguments={"player_id": "1"},
        ),
        final_reply="已查询玩家 ID 1 的资料。",
    )

    await run_customer_service_agent(
        session_id="memory-session",
        message="查询我的资料",
        llm_client=second_turn_llm,
    )

    assert second_turn_llm.decision_messages is not None
    decision_context = second_turn_llm.decision_messages[-1]["content"]
    assert "我的 ID 是 1" in decision_context
    assert "我记住了，你的 ID 是 1。" in decision_context
    assert "查询我的资料" in decision_context


async def test_llm_agent_does_not_share_history_between_sessions() -> None:
    first_turn_llm = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="记录玩家 ID",
            direct_reply="我记住了，你的 ID 是 1。",
        ),
        final_reply="不应该使用这个回复",
    )
    await run_customer_service_agent(
        session_id="memory-session-a",
        message="我的 ID 是 1",
        llm_client=first_turn_llm,
    )
    second_turn_llm = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.DIRECT_ANSWER,
            reason="普通追问",
            direct_reply="请提供你的玩家 ID。",
        ),
        final_reply="不应该使用这个回复",
    )

    await run_customer_service_agent(
        session_id="memory-session-b",
        message="查询我的资料",
        llm_client=second_turn_llm,
    )

    assert second_turn_llm.decision_messages is not None
    decision_context = second_turn_llm.decision_messages[-1]["content"]
    assert "我的 ID 是 1" not in decision_context
    assert "我记住了，你的 ID 是 1。" not in decision_context


async def test_llm_agent_falls_back_to_rules_when_decision_is_invalid() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.FALLBACK,
            reason="无法解析模型 JSON",
        ),
        final_reply="不应该使用这个回复",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        player_id="player-1",
        message="查询我的玩家资料",
        llm_client=llm_client,
    )

    assert "玩家数据查询尚未启用" in response.reply
