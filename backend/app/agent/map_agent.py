from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.agent.decision import AgentAction, AgentDecision
from app.map_tools import MapToolResult, build_map_tools

StatusEmitter = Callable[[str], None]


class MapToolsProtocol(Protocol):
    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        raise NotImplementedError

    async def geocode(self, address: str | None, *, city: str | None = None) -> MapToolResult:
        raise NotImplementedError

    async def route(
        self,
        *,
        origin: str | None,
        destination: str | None,
        mode: str | None = None,
        city: str | None = None,
        cityd: str | None = None,
    ) -> MapToolResult:
        raise NotImplementedError

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
        raise NotImplementedError

    async def weather(self, city: str | None) -> MapToolResult:
        raise NotImplementedError


@dataclass(frozen=True)
class MapAgentResult:
    decision: AgentDecision | None
    map_result: MapToolResult


async def run_map_agent(
    decision: AgentDecision | None,
    *,
    message: str,
    map_tools: MapToolsProtocol | None = None,
    emit_status: StatusEmitter | None = None,
) -> MapAgentResult:
    _emit(emit_status, "地图 Agent 正在分析地图子任务")
    tools = map_tools or build_map_tools()

    _emit(emit_status, "地图 Agent 正在调用高德地图工具")
    result = await _execute_map_decision(decision, message=message, map_tools=tools)
    return MapAgentResult(decision=decision, map_result=result)


async def _execute_map_decision(
    decision: AgentDecision | None,
    *,
    message: str,
    map_tools: MapToolsProtocol,
) -> MapToolResult:
    if decision is not None and decision.action == AgentAction.AMAP_GEO:
        return await map_tools.geocode(
            _string_argument(decision, "address"),
            city=_string_argument(decision, "city"),
        )

    if decision is not None and decision.action == AgentAction.AMAP_ROUTE:
        return await map_tools.route(
            origin=_string_argument(decision, "origin"),
            destination=_string_argument(decision, "destination"),
            mode=_string_argument(decision, "mode"),
            city=_string_argument(decision, "city"),
            cityd=_string_argument(decision, "cityd"),
        )

    if decision is not None and decision.action == AgentAction.AMAP_NAVIGATION:
        destination = _string_argument(decision, "destination")
        return await map_tools.navigation(
            destination=destination,
            destination_name=_string_argument(decision, "destination_name") or destination,
            origin=_string_argument(decision, "origin"),
            origin_name=_string_argument(decision, "origin_name"),
            mode=_string_argument(decision, "mode"),
            city=_string_argument(decision, "city"),
        )

    if decision is not None and decision.action == AgentAction.AMAP_WEATHER:
        return await map_tools.weather(_string_argument(decision, "city"))

    return await map_tools.search_place(
        _map_keywords(decision, message),
        city=_string_argument(decision, "city") if decision is not None else None,
        types=_string_argument(decision, "types") if decision is not None else None,
    )


def _map_keywords(decision: AgentDecision | None, message: str) -> str:
    if decision is not None:
        keywords = _string_argument(decision, "keywords")
        if keywords:
            return keywords
    return message


def _string_argument(decision: AgentDecision, name: str) -> str | None:
    if not decision.arguments:
        return None
    value = decision.arguments.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _emit(emit_status: StatusEmitter | None, text: str) -> None:
    if emit_status is not None:
        emit_status(text)
