import asyncio

from app.agent.customer_service import run_customer_service_agent
from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import MapAgentResult
from app.map_tools import MapToolResult, MapToolStatus
from tests.fakes import FakeLLMClient, FakeMapTools, FakeMultiPoiMapTools


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
