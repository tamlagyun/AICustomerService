from typing import Any

from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus
from app.schemas import ChatTable, ChatTableColumn


PLAYER_LIST_COLUMNS = [
    ChatTableColumn(key="player_id", label="玩家ID"),
    ChatTableColumn(key="nickname", label="昵称"),
    ChatTableColumn(key="level", label="等级"),
    ChatTableColumn(key="server_name", label="服务器"),
    ChatTableColumn(key="status", label="状态"),
    ChatTableColumn(key="desc", label="个性描述"),
]

AMAP_PLACE_COLUMNS = [
    ChatTableColumn(key="name", label="名称"),
    ChatTableColumn(key="address", label="地址"),
    ChatTableColumn(key="type", label="类型"),
    ChatTableColumn(key="distance", label="距离"),
]


def tables_for_player_data_result(
    result: PlayerDataResult,
    *,
    tool_name: str,
) -> list[ChatTable]:
    if result.status != PlayerDataStatus.FOUND or not result.data:
        return []
    if tool_name != "mysql_players_list":
        return []

    players = result.data.get("players")
    if not isinstance(players, list) or not players:
        return []

    rows = [_row_for_columns(row, PLAYER_LIST_COLUMNS) for row in players if isinstance(row, dict)]
    if not rows:
        return []

    return [
        ChatTable(
            title="玩家列表",
            columns=PLAYER_LIST_COLUMNS,
            rows=rows,
        )
    ]


def tables_for_map_result(result: MapToolResult) -> list[ChatTable]:
    if result.status != MapToolStatus.FOUND or not result.data:
        return []

    tool_name = result.data.get("tool")
    if tool_name != "maps_text_search":
        return []

    pois = _extract_pois(result.data.get("result"))
    if not pois:
        return []

    rows = [_row_for_columns(poi, AMAP_PLACE_COLUMNS) for poi in pois if isinstance(poi, dict)]
    if not rows:
        return []

    return [
        ChatTable(
            title="高德地图地点结果",
            columns=AMAP_PLACE_COLUMNS,
            rows=rows,
        )
    ]


def _extract_pois(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []

    structured_content = value.get("structuredContent")
    pois = _find_list_by_key(structured_content, "pois")
    if pois:
        return pois

    results = _find_list_by_key(structured_content, "results")
    if results:
        return results

    if isinstance(structured_content, list):
        return [item for item in structured_content if isinstance(item, dict)]

    return []


def _find_list_by_key(value: object, key: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        matched = value.get(key)
        if isinstance(matched, list):
            return [item for item in matched if isinstance(item, dict)]
        for nested in value.values():
            result = _find_list_by_key(nested, key)
            if result:
                return result
    if isinstance(value, list):
        for item in value:
            result = _find_list_by_key(item, key)
            if result:
                return result
    return []


def _row_for_columns(row: dict[str, Any], columns: list[ChatTableColumn]) -> dict[str, Any]:
    return {column.key: _display_value(row.get(column.key)) for column in columns}


def _display_value(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
