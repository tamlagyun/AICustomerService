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


class ToolArgumentType(StrEnum):
    STR = "str"
    INT = "int"
    DICT = "dict"


@dataclass(frozen=True)
class ToolArgumentDefinition:
    name: str
    argument_type: ToolArgumentType
    required: bool = False
    default: object | None = None
    min_value: int | None = None
    max_value: int | None = None
    choices: tuple[object, ...] = ()


@dataclass(frozen=True)
class ToolArgumentsValidationResult:
    valid: bool
    arguments: dict[str, object]
    errors: list[str]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    action: AgentAction
    category: ToolCategory
    description: str
    dependencies: tuple[ToolDependency, ...] = ()
    backend_entrypoint: str = ""
    external_tool_name: str = ""
    arguments_schema: tuple[ToolArgumentDefinition, ...] = ()


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
        arguments_schema=(
            ToolArgumentDefinition("player_id", ToolArgumentType.STR, required=True),
        ),
    ),
    ToolDefinition(
        name="mysql_players_list",
        action=AgentAction.MYSQL_PLAYERS_LIST,
        category=ToolCategory.MYSQL,
        description="查询 players 表玩家列表，默认 limit=100，最大 limit=1000。",
        dependencies=(ToolDependency.MYSQL,),
        backend_entrypoint="app.player_data.PlayerDataTools.get_players",
        arguments_schema=(
            ToolArgumentDefinition(
                "limit",
                ToolArgumentType.INT,
                default=100,
                min_value=1,
                max_value=1000,
            ),
        ),
    ),
    ToolDefinition(
        name="avatar_generate",
        action=AgentAction.AVATAR_GENERATE,
        category=ToolCategory.AVATAR,
        description="根据已查询到的玩家资料生成本地 PNG 个性头像。",
        dependencies=(ToolDependency.MYSQL,),
        backend_entrypoint="app.avatar_generation.AvatarGenerator.generate_player_avatar",
        arguments_schema=(
            ToolArgumentDefinition("player_id", ToolArgumentType.STR, required=True),
        ),
    ),
    ToolDefinition(
        name="amap_place_search",
        action=AgentAction.AMAP_PLACE_SEARCH,
        category=ToolCategory.MAP,
        description="调用高德 MCP 查询地点、景点、POI 信息。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.search_place",
        external_tool_name="maps_text_search",
        arguments_schema=(
            ToolArgumentDefinition("keywords", ToolArgumentType.STR),
            ToolArgumentDefinition("city", ToolArgumentType.STR),
            ToolArgumentDefinition("types", ToolArgumentType.STR),
            ToolArgumentDefinition("presentation", ToolArgumentType.DICT),
        ),
    ),
    ToolDefinition(
        name="amap_geo",
        action=AgentAction.AMAP_GEO,
        category=ToolCategory.MAP,
        description="调用高德 MCP 将地址或地名解析为经纬度。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.geocode",
        external_tool_name="maps_geo",
        arguments_schema=(
            ToolArgumentDefinition("address", ToolArgumentType.STR, required=True),
            ToolArgumentDefinition("city", ToolArgumentType.STR),
        ),
    ),
    ToolDefinition(
        name="amap_route",
        action=AgentAction.AMAP_ROUTE,
        category=ToolCategory.MAP,
        description="调用高德 MCP 查询路线、距离和出行方式。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.route",
        external_tool_name="maps_direction_*",
        arguments_schema=(
            ToolArgumentDefinition("origin", ToolArgumentType.STR, required=True),
            ToolArgumentDefinition("destination", ToolArgumentType.STR, required=True),
            ToolArgumentDefinition(
                "mode",
                ToolArgumentType.STR,
                choices=("bicycling", "driving", "transit", "walking"),
            ),
            ToolArgumentDefinition("city", ToolArgumentType.STR),
            ToolArgumentDefinition("cityd", ToolArgumentType.STR),
        ),
    ),
    ToolDefinition(
        name="amap_navigation",
        action=AgentAction.AMAP_NAVIGATION,
        category=ToolCategory.MAP,
        description="生成高德地图导航 URI，供前端或玩家打开导航。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.navigation",
        external_tool_name="amap_navigation_uri",
        arguments_schema=(
            ToolArgumentDefinition("destination", ToolArgumentType.STR, required=True),
            ToolArgumentDefinition("destination_name", ToolArgumentType.STR),
            ToolArgumentDefinition("origin", ToolArgumentType.STR),
            ToolArgumentDefinition("origin_name", ToolArgumentType.STR),
            ToolArgumentDefinition(
                "mode",
                ToolArgumentType.STR,
                choices=("bicycling", "driving", "transit", "walking"),
            ),
            ToolArgumentDefinition("city", ToolArgumentType.STR),
        ),
    ),
    ToolDefinition(
        name="amap_weather",
        action=AgentAction.AMAP_WEATHER,
        category=ToolCategory.MAP,
        description="调用高德 MCP 查询城市天气。",
        dependencies=(ToolDependency.AMAP,),
        backend_entrypoint="app.map_tools.AmapMapTools.weather",
        external_tool_name="maps_weather",
        arguments_schema=(
            ToolArgumentDefinition("city", ToolArgumentType.STR, required=True),
        ),
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


def validate_tool_arguments(
    action: AgentAction,
    arguments: dict[str, object] | None,
) -> ToolArgumentsValidationResult:
    raw_arguments = arguments if isinstance(arguments, dict) else {}
    definition = get_tool_by_action(action)
    if definition is None:
        return ToolArgumentsValidationResult(
            valid=True,
            arguments=raw_arguments,
            errors=[],
        )

    cleaned_arguments: dict[str, object] = {}
    errors: list[str] = []
    for argument_definition in definition.arguments_schema:
        if argument_definition.name not in raw_arguments:
            if argument_definition.default is not None:
                cleaned_arguments[argument_definition.name] = argument_definition.default
            elif argument_definition.required:
                errors.append(f"{argument_definition.name} is required")
            continue

        raw_value = raw_arguments[argument_definition.name]
        parsed_value, error = _coerce_argument(argument_definition, raw_value)
        if error:
            errors.append(error)
            continue
        if parsed_value is None:
            if argument_definition.default is not None:
                cleaned_arguments[argument_definition.name] = argument_definition.default
            elif argument_definition.required:
                errors.append(f"{argument_definition.name} is required")
            continue

        if argument_definition.choices and parsed_value not in argument_definition.choices:
            choices = ", ".join(str(choice) for choice in sorted(argument_definition.choices))
            errors.append(f"{argument_definition.name} must be one of: {choices}")
            continue

        cleaned_arguments[argument_definition.name] = _clamp_numeric_argument(
            argument_definition,
            parsed_value,
        )

    return ToolArgumentsValidationResult(
        valid=not errors,
        arguments=cleaned_arguments,
        errors=errors,
    )


def _coerce_tool_definition(tool: ToolDefinition | AgentAction | str) -> ToolDefinition | None:
    if isinstance(tool, ToolDefinition):
        return tool
    if isinstance(tool, AgentAction):
        return get_tool_by_action(tool)
    return get_tool_by_name(tool)


def _coerce_argument(
    definition: ToolArgumentDefinition,
    value: object,
) -> tuple[object | None, str | None]:
    if definition.argument_type == ToolArgumentType.STR:
        if isinstance(value, str):
            cleaned = value.strip()
            return (cleaned, None) if cleaned else (None, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value), None
        return None, f"{definition.name} must be a string"

    if definition.argument_type == ToolArgumentType.INT:
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.isdigit():
                return int(cleaned), None
        return None, f"{definition.name} must be an integer"

    if definition.argument_type == ToolArgumentType.DICT:
        if isinstance(value, dict):
            return value, None
        return None, f"{definition.name} must be an object"

    return None, f"{definition.name} has unsupported argument type"


def _clamp_numeric_argument(
    definition: ToolArgumentDefinition,
    value: object,
) -> object:
    if not isinstance(value, int) or isinstance(value, bool):
        return value
    if definition.min_value is not None and value < definition.min_value:
        return definition.min_value
    if definition.max_value is not None and value > definition.max_value:
        return definition.max_value
    return value
