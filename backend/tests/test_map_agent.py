from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import run_map_agent
from app.map_tools import MapToolResult, MapToolStatus


class FakeMapTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.calls.append(
            (
                "search_place",
                {"keywords": keywords, "city": city, "types": types},
            )
        )
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="place result",
            data={"tool": "maps_text_search"},
        )

    async def geocode(self, address: str | None, *, city: str | None = None) -> MapToolResult:
        self.calls.append(("geocode", {"address": address, "city": city}))
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="geo result",
            data={"tool": "maps_geo"},
        )

    async def route(
        self,
        *,
        origin: str | None,
        destination: str | None,
        mode: str | None = None,
        city: str | None = None,
        cityd: str | None = None,
    ) -> MapToolResult:
        self.calls.append(
            (
                "route",
                {
                    "origin": origin,
                    "destination": destination,
                    "mode": mode,
                    "city": city,
                    "cityd": cityd,
                },
            )
        )
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="route result",
            data={"tool": "maps_direction_driving"},
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
        self.calls.append(
            (
                "navigation",
                {
                    "destination": destination,
                    "destination_name": destination_name,
                    "origin": origin,
                    "origin_name": origin_name,
                    "mode": mode,
                    "city": city,
                },
            )
        )
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="navigation result",
            data={"tool": "amap_navigation_uri"},
        )

    async def weather(self, city: str | None) -> MapToolResult:
        self.calls.append(("weather", {"city": city}))
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="weather result",
            data={"tool": "maps_weather"},
        )


async def test_map_agent_dispatches_place_search_and_emits_statuses() -> None:
    map_tools = FakeMapTools()
    statuses: list[str] = []
    decision = AgentDecision(
        action=AgentAction.AMAP_PLACE_SEARCH,
        reason="need place data",
        arguments={"keywords": "attractions", "city": "Guangzhou", "types": "scenic"},
    )

    result = await run_map_agent(
        decision,
        message="list Guangzhou attractions",
        map_tools=map_tools,
        emit_status=statuses.append,
    )

    assert result.decision == decision
    assert result.map_result.status == MapToolStatus.FOUND
    assert map_tools.calls == [
        (
            "search_place",
            {"keywords": "attractions", "city": "Guangzhou", "types": "scenic"},
        )
    ]
    assert statuses == [
        "地图 Agent 正在分析地图子任务",
        "地图 Agent 正在调用高德地图工具",
    ]


async def test_map_agent_dispatches_route_with_controlled_arguments() -> None:
    map_tools = FakeMapTools()
    decision = AgentDecision(
        action=AgentAction.AMAP_ROUTE,
        reason="need route data",
        arguments={
            "origin": "A",
            "destination": "B",
            "mode": "walking",
            "city": "Guangzhou",
            "cityd": "Guangzhou",
        },
    )

    result = await run_map_agent(
        decision,
        message="route from A to B",
        map_tools=map_tools,
    )

    assert result.map_result.summary == "route result"
    assert map_tools.calls == [
        (
            "route",
            {
                "origin": "A",
                "destination": "B",
                "mode": "walking",
                "city": "Guangzhou",
                "cityd": "Guangzhou",
            },
        )
    ]


async def test_map_agent_falls_back_to_place_search_when_decision_is_not_map_action() -> None:
    map_tools = FakeMapTools()
    decision = AgentDecision(
        action=AgentAction.FALLBACK,
        reason="no llm map decision",
    )

    result = await run_map_agent(
        decision,
        message="nearby cafe",
        map_tools=map_tools,
    )

    assert result.decision == decision
    assert result.map_result.summary == "place result"
    assert map_tools.calls == [
        (
            "search_place",
            {"keywords": "nearby cafe", "city": None, "types": None},
        )
    ]

