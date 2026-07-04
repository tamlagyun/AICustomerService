from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.agent.decision import AgentAction
from app.config import Settings


class ToolCategory(StrEnum):
    KNOWLEDGE = "knowledge"
    MYSQL = "mysql"
    AVATAR = "avatar"
    MAP = "map"


class ToolDependency(StrEnum):
    MYSQL = "mysql"
    AMAP = "amap"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    action: AgentAction
    category: ToolCategory
    description: str
    dependencies: tuple[ToolDependency, ...] = ()
    backend_entrypoint: str = ""
    external_tool_name: str = ""


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="knowledge_base",
        action=AgentAction.KNOWLEDGE_BASE,
        category=ToolCategory.KNOWLEDGE,
        description="检索本地客服知识库，支持 doc 或向量库来源。",
        backend_entrypoint="app.knowledge_base.KnowledgeBaseSearch.search",
    ),
    ToolDefinition(
        name="mysql_player_profile",
        action=AgentAction.MYSQL_PLAYER_PROFILE,
        category=ToolCategory.MYSQL,
        description="按玩家 ID 查询 players 表中的单个玩家资料。",
        dependencies=(ToolDependency.MYSQL,),
        backend_entrypoint="app.player_data.PlayerDataTools.get_player_profile",
    ),
    ToolDefinition(
        name="mysql_players_list",
        action=AgentAction.MYSQL_PLAYERS_LIST,
        category=ToolCategory.MYSQL,
        description="查询 players 表玩家列表，默认 limit=100，最大 limit=1000。",
        dependencies=(ToolDependency.MYSQL,),
        backend_entrypoint="app.player_data.PlayerDataTools.get_players",
    ),
    ToolDefinition(
        name="avatar_generate",
        action=AgentAction.AVATAR_GENERATE,
        category=ToolCategory.AVATAR,
        description="根据已查询到的玩家资料生成本地 PNG 个性头像。",
        dependencies=(ToolDependency.MYSQL,),
        backend_entrypoint="app.avatar_generation.AvatarGenerator.generate_player_avatar",
    ),
    ToolDefinition(
        name="amap_place_search",
        action=AgentAction.AMAP_PLACE_SEARCH,
        category=ToolCategory.MAP,
        description="调用高德 MCP 查询地点、景点、POI 信息。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.search_place",
        external_tool_name="maps_text_search",
    ),
    ToolDefinition(
        name="amap_geo",
        action=AgentAction.AMAP_GEO,
        category=ToolCategory.MAP,
        description="调用高德 MCP 将地址或地名解析为经纬度。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.geocode",
        external_tool_name="maps_geo",
    ),
    ToolDefinition(
        name="amap_route",
        action=AgentAction.AMAP_ROUTE,
        category=ToolCategory.MAP,
        description="调用高德 MCP 查询路线、距离和出行方式。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.route",
        external_tool_name="maps_direction_*",
    ),
    ToolDefinition(
        name="amap_navigation",
        action=AgentAction.AMAP_NAVIGATION,
        category=ToolCategory.MAP,
        description="生成高德地图导航 URI，供前端或玩家打开导航。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.navigation",
        external_tool_name="amap_navigation_uri",
    ),
    ToolDefinition(
        name="amap_weather",
        action=AgentAction.AMAP_WEATHER,
        category=ToolCategory.MAP,
        description="调用高德 MCP 查询城市天气。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.weather",
        external_tool_name="maps_weather",
    ),
)

_TOOLS_BY_ACTION = {definition.action: definition for definition in TOOL_DEFINITIONS}
_TOOLS_BY_NAME = {definition.name: definition for definition in TOOL_DEFINITIONS}


def list_tool_definitions() -> tuple[ToolDefinition, ...]:
    return TOOL_DEFINITIONS


def get_tool_by_action(action: AgentAction) -> ToolDefinition | None:
    return _TOOLS_BY_ACTION.get(action)


def get_tool_by_name(name: str) -> ToolDefinition | None:
    return _TOOLS_BY_NAME.get(name)


def is_registered_tool_action(action: AgentAction) -> bool:
    return action in _TOOLS_BY_ACTION


def is_map_tool_action(action: AgentAction) -> bool:
    tool = get_tool_by_action(action)
    return tool is not None and tool.category == ToolCategory.MAP


def missing_tool_dependencies(
    settings: Settings,
    tool: ToolDefinition | AgentAction | str,
) -> list[ToolDependency]:
    definition = _coerce_tool_definition(tool)
    if definition is None:
        return []

    missing: list[ToolDependency] = []
    for dependency in definition.dependencies:
        if dependency == ToolDependency.MYSQL and not settings.mysql_enabled:
            missing.append(dependency)
        if dependency == ToolDependency.AMAP and not settings.amap_mcp_enabled:
            missing.append(dependency)
    return missing


def _coerce_tool_definition(tool: ToolDefinition | AgentAction | str) -> ToolDefinition | None:
    if isinstance(tool, ToolDefinition):
        return tool
    if isinstance(tool, AgentAction):
        return get_tool_by_action(tool)
    return get_tool_by_name(tool)
