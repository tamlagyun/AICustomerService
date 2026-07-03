from dataclasses import dataclass
import re
from typing import Any, Literal

from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus
from app.schemas import ChatTable, ChatTableColumn

PresentationMode = Literal["auto", "table", "text"]
SortOrder = Literal["asc", "desc"]


@dataclass(frozen=True)
class PresentationPlan:
    mode: PresentationMode = "auto"
    sort_by: str | None = None
    sort_order: SortOrder = "asc"
    columns: tuple[str, ...] = ()


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

PLAYER_ATTRACTION_RECOMMENDATION_COLUMNS = [
    ChatTableColumn(key="player_id", label="玩家ID"),
    ChatTableColumn(key="nickname", label="昵称"),
    ChatTableColumn(key="desc", label="个性描述"),
    ChatTableColumn(key="recommended_place", label="推荐景点"),
    ChatTableColumn(key="recommendation_reason", label="推荐原因"),
    ChatTableColumn(key="expected_mood", label="游玩后可能收获"),
]

TEXT_MODE_PATTERNS = [
    "不用表格",
    "不要表格",
    "别用表格",
    "不需要表格",
    "无需表格",
    "用文字",
    "文字说明",
    "文本说明",
]

TABLE_MODE_PATTERNS = [
    "用表格",
    "表格显示",
    "表格排列",
    "表格列出",
    "做成表格",
    "生成表格",
    "以表格",
]

LIST_INTENT_PATTERNS = [
    "列出",
    "所有",
    "全部",
    "列表",
    "清单",
    "排行",
    "排名",
    "排序",
    "对比",
    "有哪些",
    "分别",
]


def build_presentation_plan(
    message: str,
    *,
    conversation_history: list[object] | None = None,
    decision_arguments: dict[str, object] | None = None,
) -> PresentationPlan:
    """Build a deterministic display plan from user text, history and LLM arguments."""
    text = message.strip()
    lower_text = text.lower()

    if _contains_any(text, TEXT_MODE_PATTERNS):
        mode: PresentationMode = "text"
    else:
        mode = _mode_from_decision_arguments(decision_arguments)
        if mode == "auto" and _contains_any(text, TABLE_MODE_PATTERNS):
            mode = "table"
        if mode == "auto" and _looks_like_structured_list_request(text):
            mode = "table"
        if mode == "auto" and _history_prefers_table(conversation_history):
            mode = "table"

    sort_by = _sort_by_from_decision_arguments(decision_arguments)
    if not sort_by:
        sort_by = _infer_sort_by(text, lower_text)

    sort_order = _sort_order_from_decision_arguments(decision_arguments)
    if sort_order == "asc" and _looks_like_descending_sort(text, lower_text):
        sort_order = "desc"
    if sort_by == "level" and _looks_like_rank_request(text):
        sort_order = "desc"

    return PresentationPlan(
        mode=mode,
        sort_by=sort_by,
        sort_order=sort_order,
        columns=_columns_from_decision_arguments(decision_arguments),
    )


def tables_for_player_data_result(
    result: PlayerDataResult,
    *,
    tool_name: str,
    presentation_plan: PresentationPlan | None = None,
) -> list[ChatTable]:
    if result.status != PlayerDataStatus.FOUND or not result.data:
        return []
    if tool_name != "mysql_players_list":
        return []

    players = result.data.get("players")
    if not isinstance(players, list) or not players:
        return []

    plan = presentation_plan or PresentationPlan()
    if plan.mode == "text":
        return []

    raw_rows = [row for row in players if isinstance(row, dict)]
    raw_rows = _sort_rows(raw_rows, plan)
    columns = _columns_for_plan(PLAYER_LIST_COLUMNS, plan)
    rows = [_row_for_columns(row, columns) for row in raw_rows]
    if not rows:
        return []

    return [
        ChatTable(
            title="玩家列表",
            columns=columns,
            rows=rows,
        )
    ]


def tables_for_map_result(
    result: MapToolResult,
    *,
    presentation_plan: PresentationPlan | None = None,
) -> list[ChatTable]:
    if result.status != MapToolStatus.FOUND or not result.data:
        return []

    tool_name = result.data.get("tool")
    if tool_name != "maps_text_search":
        return []

    pois = _extract_pois(result.data.get("result"))
    if not pois:
        return []

    plan = presentation_plan or PresentationPlan()
    if plan.mode == "text":
        return []

    raw_rows = [poi for poi in pois if isinstance(poi, dict)]
    raw_rows = _sort_rows(raw_rows, plan)
    columns = _columns_for_plan(AMAP_PLACE_COLUMNS, plan)
    rows = [_row_for_columns(poi, columns) for poi in raw_rows]
    if not rows:
        return []

    return [
        ChatTable(
            title="高德地图地点结果",
            columns=columns,
            rows=rows,
        )
    ]


def tables_for_player_attraction_recommendation(
    player_result: PlayerDataResult,
    map_result: MapToolResult,
    *,
    presentation_plan: PresentationPlan | None = None,
) -> list[ChatTable]:
    if player_result.status != PlayerDataStatus.FOUND or not player_result.data:
        return []
    if map_result.status != MapToolStatus.FOUND or not map_result.data:
        return []

    plan = presentation_plan or PresentationPlan()
    if plan.mode == "text":
        return []

    players = _extract_players(player_result.data)
    pois = _extract_pois(map_result.data.get("result"))
    if not players or not pois:
        return []

    rows = []
    for index, player in enumerate(players):
        poi = _best_poi_for_player(player, pois, fallback_index=index)
        rows.append(_recommendation_row(player, poi))

    return [
        ChatTable(
            title="玩家景点推荐",
            columns=PLAYER_ATTRACTION_RECOMMENDATION_COLUMNS,
            rows=rows,
        )
    ]


def _extract_players(data: dict[str, Any]) -> list[dict[str, Any]]:
    players = data.get("players")
    if isinstance(players, list):
        return [player for player in players if isinstance(player, dict)]

    if data.get("player_id") is not None:
        return [data]

    return []


def _recommendation_row(player: dict[str, Any], poi: dict[str, Any]) -> dict[str, Any]:
    desc = _display_value(player.get("desc")) or "暂无"
    poi_name = _display_value(poi.get("name")) or "未知景点"
    poi_type = _display_value(poi.get("type")) or "景点"
    return {
        "player_id": _display_value(player.get("player_id")),
        "nickname": _display_value(player.get("nickname")),
        "desc": desc,
        "recommended_place": poi_name,
        "recommendation_reason": _recommendation_reason(str(desc), str(poi_name), str(poi_type)),
        "expected_mood": _expected_mood(str(desc)),
    }


def _best_poi_for_player(
    player: dict[str, Any],
    pois: list[dict[str, Any]],
    *,
    fallback_index: int,
) -> dict[str, Any]:
    style = _player_style(player.get("desc"))
    scored = [(_poi_score_for_style(poi, style), index, poi) for index, poi in enumerate(pois)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored and scored[0][0] > 0:
        return scored[0][2]
    return pois[fallback_index % len(pois)]


def _player_style(desc: object) -> str:
    text = str(desc or "")
    if any(keyword in text for keyword in ["进攻", "竞技", "挑战", "冲锋", "战斗"]):
        return "challenge"
    if any(keyword in text for keyword in ["探索", "研究", "策略", "机制", "收集"]):
        return "explore"
    if any(keyword in text for keyword in ["社交", "组队", "休闲", "陪伴"]):
        return "social"
    return "general"


def _poi_score_for_style(poi: dict[str, Any], style: str) -> int:
    text = f"{poi.get('name', '')} {poi.get('type', '')} {poi.get('address', '')}"
    keywords_by_style = {
        "challenge": ["塔", "山", "长隆", "乐园", "高空", "运动", "主题", "挑战"],
        "explore": ["博物馆", "纪念馆", "文化", "历史", "古", "馆", "遗址", "公园"],
        "social": ["街", "广场", "商圈", "乐园", "公园", "步行街"],
        "general": ["景区", "景点", "公园", "风景"],
    }
    return sum(1 for keyword in keywords_by_style[style] if keyword in text)


def _recommendation_reason(desc: str, poi_name: str, poi_type: str) -> str:
    style = _player_style(desc)
    if style == "challenge":
        return f"{desc}适合节奏更强、目标感更明确的游玩体验，{poi_name}这类{poi_type}更容易带来挑战感。"
    if style == "explore":
        return f"{desc}适合信息量更丰富、可慢慢观察的地点，{poi_name}这类{poi_type}更容易满足探索欲。"
    if style == "social":
        return f"{desc}适合轻松互动和结伴出行，{poi_name}这类{poi_type}更容易形成共同话题。"
    return f"根据当前个性描述，{poi_name}这类{poi_type}适合作为轻松游玩的选择。"


def _expected_mood(desc: str) -> str:
    style = _player_style(desc)
    if style == "challenge":
        return "可能收获完成挑战后的兴奋感、释放压力后的畅快感。"
    if style == "explore":
        return "可能收获好奇心被满足后的新鲜感、发现细节后的充实感。"
    if style == "social":
        return "可能收获结伴游玩的放松感、交流互动后的愉快感。"
    return "可能收获放松、愉悦和短暂抽离日常压力的心情。"


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


def _mode_from_decision_arguments(arguments: dict[str, object] | None) -> PresentationMode:
    presentation = _presentation_argument(arguments)
    raw_mode = presentation.get("mode") or presentation.get("format") or presentation.get("display")
    if isinstance(raw_mode, str):
        normalized = raw_mode.strip().lower()
        if normalized in {"table", "tables", "grid"}:
            return "table"
        if normalized in {"text", "plain_text", "paragraph"}:
            return "text"
    return "auto"


def _sort_by_from_decision_arguments(arguments: dict[str, object] | None) -> str | None:
    presentation = _presentation_argument(arguments)
    value = presentation.get("sort_by") or presentation.get("order_by")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _sort_order_from_decision_arguments(arguments: dict[str, object] | None) -> SortOrder:
    presentation = _presentation_argument(arguments)
    value = presentation.get("sort_order") or presentation.get("order")
    if isinstance(value, str) and value.strip().lower() == "desc":
        return "desc"
    return "asc"


def _columns_from_decision_arguments(arguments: dict[str, object] | None) -> tuple[str, ...]:
    presentation = _presentation_argument(arguments)
    raw_columns = presentation.get("columns")
    if not isinstance(raw_columns, list):
        return ()
    columns = [column.strip() for column in raw_columns if isinstance(column, str) and column.strip()]
    return tuple(dict.fromkeys(columns))


def _presentation_argument(arguments: dict[str, object] | None) -> dict[str, object]:
    if not arguments:
        return {}
    value = arguments.get("presentation")
    if isinstance(value, dict):
        return value
    return {}


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _looks_like_structured_list_request(text: str) -> bool:
    return _contains_any(text, LIST_INTENT_PATTERNS)


def _history_prefers_table(conversation_history: list[object] | None) -> bool:
    if not conversation_history:
        return False
    for message in conversation_history[-4:]:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        if _contains_any(content, TEXT_MODE_PATTERNS):
            return False
        if any(keyword in content for keyword in ["以后都用表格", "后续都用表格", "接下来都用表格"]):
            return True
    return False


def _infer_sort_by(text: str, lower_text: str) -> str | None:
    if "distance" in lower_text or "距离" in text or "最近" in text:
        return "distance"
    if "level" in lower_text or "等级" in text:
        return "level"
    if "player_id" in lower_text or "玩家id" in lower_text or "id" == lower_text:
        return "player_id"
    if "desc" in lower_text or ("个性" in text and "分类" in text):
        return "desc"
    return None


def _looks_like_descending_sort(text: str, lower_text: str) -> bool:
    return any(
        keyword in text or keyword in lower_text
        for keyword in ["降序", "由高到低", "从高到低", "由远到近", "从远到近"]
    )


def _looks_like_rank_request(text: str) -> bool:
    return any(keyword in text for keyword in ["排名", "排行", "最高", "由高到低", "从高到低"])


def _columns_for_plan(
    default_columns: list[ChatTableColumn],
    plan: PresentationPlan,
) -> list[ChatTableColumn]:
    if not plan.columns:
        return default_columns

    default_by_key = {column.key: column for column in default_columns}
    columns: list[ChatTableColumn] = []
    for key in plan.columns:
        columns.append(default_by_key.get(key, ChatTableColumn(key=key, label=key)))
    return columns


def _sort_rows(rows: list[dict[str, Any]], plan: PresentationPlan) -> list[dict[str, Any]]:
    if not plan.sort_by:
        return rows

    keyed_rows = [(row, _sortable_value(row.get(plan.sort_by), plan.sort_by)) for row in rows]
    present_rows = [(row, value) for row, value in keyed_rows if value is not None]
    missing_rows = [row for row, value in keyed_rows if value is None]
    if not present_rows:
        return rows

    present_rows.sort(key=lambda item: item[1], reverse=plan.sort_order == "desc")
    return [row for row, _ in present_rows] + missing_rows


def _sortable_value(value: object, key: str) -> tuple[int, float | str] | None:
    if value is None:
        return None
    if key == "distance":
        distance = _distance_to_meters(value)
        return (0, distance) if distance is not None else None
    if isinstance(value, int | float):
        return (0, float(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        numeric = _string_to_float(stripped)
        if numeric is not None:
            return (0, numeric)
        return (1, stripped.lower())
    return (1, str(value).lower())


def _distance_to_meters(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    distance = float(match.group(1))
    if "km" in text or "公里" in text or "千米" in text:
        return distance * 1000
    return distance


def _string_to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
