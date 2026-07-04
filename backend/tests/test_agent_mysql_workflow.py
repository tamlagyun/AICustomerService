from app.agent.customer_service import run_customer_service_agent
from app.agent.decision import AgentAction, AgentDecision
from app.config import get_settings
from app.conversation_memory import get_conversation_memory
from tests.fakes import FakeAvatarGenerator, FakeLLMClient, FakePlayerDataTools


async def test_llm_agent_uses_mysql_player_profile_action() -> None:
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYER_PROFILE,
            reason="玩家要查资料",
        ),
        final_reply="玩家数据查询尚未启用，请配置 MySQL 后再查询。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        player_id="player-1",
        message="查询我的玩家资料",
        llm_client=llm_client,
    )

    assert "玩家数据查询尚未启用" in response.reply
    assert response.sources == []
    assert llm_client.final_messages is not None
    assert "玩家数据查询尚未启用" in llm_client.final_messages[-1]["content"]


async def test_llm_agent_uses_decision_arguments_and_desc_for_final_analysis(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYER_PROFILE,
            reason="玩家要求查资料并总结个性",
            arguments={"player_id": "1"},
            final_task="根据玩家资料和 desc 字段总结玩家个性",
        ),
        final_reply="根据资料和个性描述，你偏好研究机制，适合循序渐进体验。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="根据ID=1查询我的资料并对我的资料进行分析总结出我的个性",
        llm_client=llm_client,
    )

    assert player_tools.requested_player_id == "1"
    assert response.reply == "根据资料和个性描述，你偏好研究机制，适合循序渐进体验。"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "根据玩家资料和 desc 字段总结玩家个性" in final_prompt
    assert "喜欢研究机制" in final_prompt


async def test_llm_agent_truncates_long_history_but_keeps_question_and_tool_result(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_CONTEXT_MAX_TOKENS", "2400")
    monkeypatch.setenv("LLM_CONTEXT_RESERVED_REPLY_TOKENS", "200")
    get_settings.cache_clear()
    session_id = "context-budget-session"
    memory = get_conversation_memory()
    memory.clear_session(session_id)
    for index in range(5):
        memory.append_message(session_id, "user", f"旧历史-{index}-" + "很长的旧对话" * 200)
        memory.append_message(session_id, "assistant", f"旧回复-{index}-" + "很长的旧回复" * 200)

    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYER_PROFILE,
            reason="玩家要求查资料并总结个性",
            arguments={"player_id": "1"},
            final_task="根据玩家资料和 desc 字段总结玩家个性",
        ),
        final_reply="根据资料和个性描述，你偏好研究机制。",
    )

    try:
        await run_customer_service_agent(
            session_id=session_id,
            message="根据ID=1查询我的资料并分析我的个性",
            llm_client=llm_client,
        )
    finally:
        get_settings.cache_clear()
        memory.clear_session(session_id)

    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "根据ID=1查询我的资料并分析我的个性" in final_prompt
    assert "喜欢研究机制" in final_prompt
    assert "旧历史" not in final_prompt


async def test_llm_agent_asks_clarification_without_calling_mysql(monkeypatch) -> None:
    did_build_tools = False

    def build_tools():
        nonlocal did_build_tools
        did_build_tools = True
        return FakePlayerDataTools()

    monkeypatch.setattr("app.agent.customer_service.build_player_data_tools", build_tools)
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.ASK_CLARIFICATION,
            reason="缺少玩家 ID",
            arguments={"missing": ["player_id"]},
            direct_reply="请提供玩家 ID，我才能查询资料。",
        ),
        final_reply="不应该生成最终回复",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="帮我查资料并总结个性",
        llm_client=llm_client,
    )

    assert response.reply == "请提供玩家 ID，我才能查询资料。"
    assert did_build_tools is False
    assert llm_client.final_messages is None


async def test_llm_agent_uses_players_list_action_and_limit(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.MYSQL_PLAYERS_LIST,
            reason="用户要求查询 players 表所有数据",
            arguments={"limit": 1000},
            final_task="总结当前玩家列表整体情况",
        ),
        final_reply="当前玩家中包含进攻型和探索型玩家。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="查询数据库players表中所有数据，并总结整体情况",
        llm_client=llm_client,
    )

    assert player_tools.requested_limit == 1000
    assert response.reply == "当前玩家中包含进攻型和探索型玩家。"
    assert len(response.tables) == 1
    assert response.tables[0].title == "玩家列表"
    assert response.tables[0].rows[0]["nickname"] == "ai大名"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "mysql_players_list" in final_prompt
    assert "进攻型玩家" in final_prompt


async def test_local_rules_route_database_all_profiles_to_players_list(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="查询数据库中所有的资料",
        llm_client=None,
    )

    assert player_tools.requested_limit == 100
    assert "共查询到 2 条玩家数据" in response.reply
    assert len(response.tables) == 1


async def test_llm_agent_generates_avatar_from_player_profile(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    avatar_generator = FakeAvatarGenerator()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )
    monkeypatch.setattr(
        "app.agent.customer_service.build_avatar_generator",
        lambda: avatar_generator,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.AVATAR_GENERATE,
            reason="玩家要求根据资料生成头像",
            arguments={"player_id": "1"},
            final_task="根据玩家资料生成符合个性的头像",
        ),
        final_reply="头像已生成，风格偏策略研究型。",
    )

    response = await run_customer_service_agent(
        session_id="avatar-session",
        message="player_id=1根据我的资料生成头像",
        llm_client=llm_client,
    )

    assert player_tools.requested_player_id == "1"
    assert avatar_generator.session_id == "avatar-session"
    assert avatar_generator.profile is not None
    assert avatar_generator.profile["desc"] == "喜欢研究机制。"
    assert response.images
    assert response.images[0].url == "/generated/avatars/player-1.png"
    assert response.reply == "头像已生成，风格偏策略研究型。"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "avatar_generate" in final_prompt
    assert "/generated/avatars/player-1.png" in final_prompt
