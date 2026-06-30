from app.agent.customer_service import run_customer_service_agent
from app.agent.decision import AgentAction, AgentDecision
from app.llm import LLMResponse
from app.player_data import PlayerDataResult, PlayerDataStatus


class FakeLLMClient:
    def __init__(self, *, decision: AgentDecision, final_reply: str) -> None:
        self.decision = decision
        self.final_reply = final_reply
        self.decision_messages: list[dict[str, str]] | None = None
        self.final_messages: list[dict[str, str]] | None = None

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        self.decision_messages = messages
        return self.decision

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.final_messages = messages
        return LLMResponse(content=self.final_reply)


class FakePlayerDataTools:
    def __init__(self) -> None:
        self.requested_player_id: str | None = None
        self.requested_limit: int | None = None

    def get_player_profile(self, player_id: str | None) -> PlayerDataResult:
        self.requested_player_id = player_id
        return PlayerDataResult(
            status=PlayerDataStatus.FOUND,
            summary="玩家资料：玩家 ID 1，昵称 ai大名，等级 2，服务器 1服，状态 1，个性描述 喜欢研究机制。",
            data={
                "player_id": "1",
                "nickname": "ai大名",
                "level": 2,
                "server_name": "1服",
                "status": "1",
                "desc": "喜欢研究机制。",
            },
        )

    def get_players(self, limit: int = 100) -> PlayerDataResult:
        self.requested_limit = limit
        return PlayerDataResult(
            status=PlayerDataStatus.FOUND,
            summary="共查询到 2 条玩家数据，当前返回上限 100 条。",
            data={
                "limit": limit,
                "players": [
                    {
                        "player_id": "1",
                        "nickname": "ai大名",
                        "level": 2,
                        "server_name": "1服",
                        "status": "1",
                        "desc": "进攻型玩家",
                    },
                    {
                        "player_id": "2",
                        "nickname": "beta",
                        "level": 8,
                        "server_name": "1服",
                        "status": "1",
                        "desc": "探索型玩家",
                    },
                ],
            },
        )


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
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "mysql_players_list" in final_prompt
    assert "进攻型玩家" in final_prompt


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
