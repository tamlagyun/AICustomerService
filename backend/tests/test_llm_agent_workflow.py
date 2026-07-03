import asyncio
import json
import logging

import pytest

from app.agent.customer_service import run_customer_service_agent, stream_customer_service_agent
from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import MapAgentResult
from app.avatar_generation import AvatarGenerationResult, AvatarGenerationStatus
from app.config import get_settings
from app.llm import LLMResponse
from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus
from app.prompt_registry import PromptNotFoundError


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


class MultiStepFakeLLMClient:
    def __init__(self, *, decisions: list[AgentDecision], final_reply: str) -> None:
        self.decisions = decisions
        self.final_reply = final_reply
        self.decision_messages: list[list[dict[str, str]]] = []
        self.final_messages: list[dict[str, str]] | None = None

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        self.decision_messages.append(messages)
        return self.decisions.pop(0)

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.final_messages = messages
        return LLMResponse(content=self.final_reply)


class PlannerFakeLLMClient:
    def __init__(
        self,
        *,
        plan_reply: str,
        final_reply: str,
        fallback_decision: AgentDecision | None = None,
    ) -> None:
        self.plan_reply = plan_reply
        self.final_reply = final_reply
        self.fallback_decision = fallback_decision
        self.decision_messages: list[list[dict[str, str]]] = []
        self.generate_messages: list[list[dict[str, str]]] = []

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        self.decision_messages.append(messages)
        return self.fallback_decision or AgentDecision(
            action=AgentAction.FALLBACK,
            reason="Planner test should not use decide_action",
        )

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.generate_messages.append(messages)
        if len(self.generate_messages) == 1:
            return LLMResponse(content=self.plan_reply)
        return LLMResponse(content=self.final_reply)


class FailingDecisionLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        raise TimeoutError("llm timeout")

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(content="不应调用最终生成")


class FailingFinalReplyLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return AgentDecision(
            action=AgentAction.MYSQL_PLAYERS_LIST,
            reason="查询所有玩家资料",
        )

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        raise TimeoutError("llm final timeout")


class FailingStreamLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return AgentDecision(
            action=AgentAction.MYSQL_PLAYERS_LIST,
            reason="查询所有玩家资料",
        )

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(content="流式失败后使用普通生成回复")

    async def stream_reply(self, messages: list[dict[str, str]]):
        raise TimeoutError("llm stream timeout")
        yield ""


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


class FakeAvatarGenerator:
    def __init__(self) -> None:
        self.profile: dict | None = None
        self.session_id: str | None = None

    def generate_player_avatar(
        self,
        profile: dict,
        *,
        session_id: str,
    ) -> AvatarGenerationResult:
        self.profile = profile
        self.session_id = session_id
        return AvatarGenerationResult(
            status=AvatarGenerationStatus.GENERATED,
            summary="已生成本地 PNG 头像：/generated/avatars/player-1.png",
            url="/generated/avatars/player-1.png",
            alt="ai大名 的个性头像",
            data={"style": "策略研究型"},
        )


class FakeMapTools:
    def __init__(self) -> None:
        self.place_query: dict[str, object] | None = None

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.place_query = {"keywords": keywords, "city": city, "types": types}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图查询结果：中关村电竞网吧，地址 北京市海淀区中关村大街 1 号。",
            data={
                "tool": "maps_text_search",
                "arguments": {"keywords": keywords, "city": city, "types": types},
                "result": {
                    "structuredContent": {
                        "pois": [
                            {
                                "name": "中关村电竞网吧",
                                "address": "北京市海淀区中关村大街 1 号",
                                "type": "休闲娱乐",
                                "distance": "800",
                            }
                        ]
                    },
                    "content": [
                        {
                            "type": "text",
                            "text": "中关村电竞网吧，地址 北京市海淀区中关村大街 1 号。",
                        }
                    ]
                },
            },
        )

    async def navigation(
        self,
        *,
        destination: str | None,
        destination_name: str | None = None,
        origin: str | None = None,
        origin_name: str | None = None,
        mode: str | None = None,
        city: str | None = None,
    ) -> MapToolResult:
        self.place_query = {
            "destination": destination,
            "destination_name": destination_name,
            "origin": origin,
            "origin_name": origin_name,
            "mode": mode,
            "city": city,
        }
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图导航链接：https://uri.amap.com/navigation?to=116.397,39.908,天安门&mode=walk",
            data={
                "tool": "amap_navigation_uri",
                "url": "https://uri.amap.com/navigation?to=116.397,39.908,天安门&mode=walk",
            },
        )

    async def weather(self, city: str | None) -> MapToolResult:
        self.place_query = {"city": city}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图天气查询结果：北京今天晴，气温 26 到 34 度。",
            data={
                "tool": "maps_weather",
                "arguments": {"city": city},
                "result": {
                    "content": [{"type": "text", "text": "北京今天晴，气温 26 到 34 度。"}]
                },
            },
        )


class FakeMultiPoiMapTools:
    def __init__(self) -> None:
        self.place_query: dict[str, object] | None = None

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.place_query = {"keywords": keywords, "city": city, "types": types}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图查询结果：广州景点列表。",
            data={
                "tool": "maps_text_search",
                "arguments": {"keywords": keywords, "city": city, "types": types},
                "result": {
                    "structuredContent": {
                        "pois": [
                            {
                                "name": "较远景点",
                                "address": "广州市越秀区",
                                "type": "风景名胜",
                                "distance": "2km",
                            },
                            {
                                "name": "较近景点",
                                "address": "广州市天河区",
                                "type": "风景名胜",
                                "distance": "120",
                            },
                            {
                                "name": "中等景点",
                                "address": "广州市海珠区",
                                "type": "风景名胜",
                                "distance": "800米",
                            },
                        ]
                    },
                    "content": [{"type": "text", "text": "广州景点列表。"}],
                },
            },
        )


class FakeGuangzhouAttractionMapTools:
    def __init__(self) -> None:
        self.place_query: dict[str, object] | None = None

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.place_query = {"keywords": keywords, "city": city, "types": types}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图查询结果：广州塔、广东省博物馆。",
            data={
                "tool": "maps_text_search",
                "arguments": {"keywords": keywords, "city": city, "types": types},
                "result": {
                    "structuredContent": {
                        "pois": [
                            {
                                "name": "广州塔",
                                "address": "广州市海珠区阅江西路222号",
                                "type": "风景名胜",
                                "distance": "600",
                            },
                            {
                                "name": "广东省博物馆",
                                "address": "广州市天河区珠江东路2号",
                                "type": "科教文化服务",
                                "distance": "900",
                            },
                        ]
                    },
                    "content": [{"type": "text", "text": "广州塔、广东省博物馆。"}],
                },
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
    assert len(response.tables) == 1
    assert response.tables[0].title == "玩家列表"
    assert response.tables[0].rows[0]["nickname"] == "ai大名"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "mysql_players_list" in final_prompt
    assert "进攻型玩家" in final_prompt


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


async def test_local_rules_route_database_all_profiles_to_players_list(monkeypatch) -> None:
    player_tools = FakePlayerDataTools()
    monkeypatch.setattr(
        "app.agent.customer_service.build_player_data_tools",
        lambda: player_tools,
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="查询数据库中所有的资料并且根据desc进行分类，用表格显示出来",
    )

    assert player_tools.requested_limit == 100
    assert "基础客服 Agent" not in response.reply
    assert len(response.tables) == 1
    assert response.tables[0].title == "玩家列表"
    assert {row["desc"] for row in response.tables[0].rows} == {"进攻型玩家", "探索型玩家"}


async def test_agent_emits_status_when_llm_decision_fails_then_uses_local_players_list(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(logging.ERROR, logger="app.agent.customer_service")
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
        final_reply="已根据你的资料生成了一个策略研究型头像。",
    )

    response = await run_customer_service_agent(
        session_id="avatar-session",
        message="根据ID=1查询我的资料并生成符合个性的头像",
        llm_client=llm_client,
    )

    assert player_tools.requested_player_id == "1"
    assert avatar_generator.session_id == "avatar-session"
    assert avatar_generator.profile is not None
    assert avatar_generator.profile["desc"] == "喜欢研究机制。"
    assert response.reply == "已根据你的资料生成了一个策略研究型头像。"
    assert len(response.images) == 1
    assert response.images[0].url == "/generated/avatars/player-1.png"
    assert response.images[0].alt == "ai大名 的个性头像"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "avatar_generate" in final_prompt
    assert "/generated/avatars/player-1.png" in final_prompt


async def test_llm_agent_uses_amap_place_search_action(monkeypatch) -> None:
    map_tools = FakeMapTools()
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.AMAP_PLACE_SEARCH,
            reason="玩家询问附近地点",
            arguments={"keywords": "网吧", "city": "北京"},
            final_task="回答玩家可选择的附近地点",
        ),
        final_reply="可以考虑中关村电竞网吧，地址在北京市海淀区中关村大街 1 号。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="北京附近有没有网吧？",
        llm_client=llm_client,
    )

    assert map_tools.place_query == {"keywords": "网吧", "city": "北京", "types": None}
    assert response.reply == "可以考虑中关村电竞网吧，地址在北京市海淀区中关村大街 1 号。"
    assert response.sources[0].source_type == "amap_mcp"
    assert response.sources[0].reference == "maps_text_search"
    assert len(response.tables) == 1
    assert response.tables[0].title == "高德地图地点结果"
    assert response.tables[0].rows[0]["name"] == "中关村电竞网吧"
    assert llm_client.final_messages is not None
    final_prompt = llm_client.final_messages[-1]["content"]
    assert "高德地图查询结果" in final_prompt
    assert "maps_text_search" in final_prompt


async def test_customer_service_delegates_map_work_to_map_agent(monkeypatch) -> None:
    delegated: dict[str, object] = {}
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    decision = AgentDecision(
        action=AgentAction.AMAP_PLACE_SEARCH,
        reason="玩家询问附近地点",
        arguments={"keywords": "网吧", "city": "北京"},
        final_task="回答玩家可选择的附近地点",
    )

    async def fake_run_map_agent(
        received_decision: AgentDecision | None,
        *,
        message: str,
        emit_status,
    ) -> MapAgentResult:
        delegated["decision"] = received_decision
        delegated["message"] = message
        emit_status("地图 Agent 正在分析地图子任务")
        emit_status("地图 Agent 正在调用高德地图工具")
        return MapAgentResult(
            decision=received_decision,
            map_result=MapToolResult(
                status=MapToolStatus.FOUND,
                summary="高德地图查询结果：中关村电竞网吧，地址 北京市海淀区中关村大街 1 号。",
                data={
                    "tool": "maps_text_search",
                    "arguments": {"keywords": "网吧", "city": "北京", "types": None},
                    "result": {
                        "structuredContent": {
                            "pois": [
                                {
                                    "name": "中关村电竞网吧",
                                    "address": "北京市海淀区中关村大街 1 号",
                                    "type": "休闲娱乐",
                                    "distance": "800",
                                }
                            ]
                        }
                    },
                },
            ),
        )

    monkeypatch.setattr("app.agent.customer_service.run_map_agent", fake_run_map_agent)
    llm_client = FakeLLMClient(
        decision=decision,
        final_reply="可以考虑中关村电竞网吧。",
    )

    response = await run_customer_service_agent(
        session_id="session-map-agent",
        message="北京附近网吧",
        llm_client=llm_client,
        status_queue=status_queue,
    )

    statuses = []
    while not status_queue.empty():
        statuses.append(status_queue.get_nowait())

    assert delegated == {"decision": decision, "message": "北京附近网吧"}
    assert "正在委托地图 Agent" in statuses
    assert "地图 Agent 正在分析地图子任务" in statuses
    assert "地图 Agent 正在调用高德地图工具" in statuses
    assert response.reply == "可以考虑中关村电竞网吧。"
    assert response.sources[0].reference == "maps_text_search"
    assert response.tables[0].title == "高德地图地点结果"


async def test_llm_agent_applies_presentation_plan_to_map_table(monkeypatch) -> None:
    map_tools = FakeMultiPoiMapTools()
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.AMAP_PLACE_SEARCH,
            reason="玩家要求列出广州旅游景点",
            arguments={
                "keywords": "旅游景点",
                "city": "广州",
                "presentation": {"mode": "table", "sort_by": "distance", "sort_order": "asc"},
            },
            final_task="按表格列出广州旅游景点，并按距离排序",
        ),
        final_reply="已按距离整理广州旅游景点。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="列出广州所有的旅游景点，用表格按距离排序",
        llm_client=llm_client,
    )

    assert map_tools.place_query == {"keywords": "旅游景点", "city": "广州", "types": None}
    assert len(response.tables) == 1
    assert [row["name"] for row in response.tables[0].rows] == ["较近景点", "中等景点", "较远景点"]
    assert llm_client.final_messages is not None
    assert "结构化表格已经由后端生成" in llm_client.final_messages[0]["content"]


async def test_llm_agent_respects_text_presentation_for_map_result(monkeypatch) -> None:
    map_tools = FakeMultiPoiMapTools()
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.AMAP_PLACE_SEARCH,
            reason="玩家要求文字介绍广州旅游景点",
            arguments={"keywords": "旅游景点", "city": "广州"},
        ),
        final_reply="广州有多个旅游景点，可以根据区域和出行距离选择。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="列出广州所有的旅游景点，不用表格，用文字说明",
        llm_client=llm_client,
    )

    assert response.tables == []


async def test_llm_agent_uses_amap_navigation_action(monkeypatch) -> None:
    map_tools = FakeMapTools()
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.AMAP_NAVIGATION,
            reason="玩家要求导航到目的地",
            arguments={"destination": "天安门", "city": "北京", "mode": "walking"},
            final_task="提供高德地图导航链接",
        ),
        final_reply="这是高德地图导航链接：https://uri.amap.com/navigation?to=116.397,39.908,天安门&mode=walk",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="导航到北京天安门",
        llm_client=llm_client,
    )

    assert map_tools.place_query == {
        "destination": "天安门",
        "destination_name": "天安门",
        "origin": None,
        "origin_name": None,
        "mode": "walking",
        "city": "北京",
    }
    assert "高德地图导航链接" in response.reply
    assert response.sources[0].reference == "amap_navigation_uri"
    assert llm_client.final_messages is not None
    assert "amap_navigation_uri" in llm_client.final_messages[-1]["content"]


async def test_llm_agent_uses_amap_weather_action(monkeypatch) -> None:
    map_tools = FakeMapTools()
    monkeypatch.setattr(
        "app.agent.map_agent.build_map_tools",
        lambda: map_tools,
    )
    llm_client = FakeLLMClient(
        decision=AgentDecision(
            action=AgentAction.AMAP_WEATHER,
            reason="玩家询问天气",
            arguments={"city": "北京"},
        ),
        final_reply="北京今天晴，气温 26 到 34 度，适合出行。",
    )

    response = await run_customer_service_agent(
        session_id="session-1",
        message="北京今天天气怎么样？",
        llm_client=llm_client,
    )

    assert map_tools.place_query == {"city": "北京"}
    assert response.reply == "北京今天晴，气温 26 到 34 度，适合出行。"
    assert response.sources[0].reference == "maps_weather"
    assert llm_client.final_messages is not None
    assert "maps_weather" in llm_client.final_messages[-1]["content"]


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
