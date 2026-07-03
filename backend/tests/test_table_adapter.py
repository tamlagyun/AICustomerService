from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus
from app.table_adapter import (
    PresentationPlan,
    build_presentation_plan,
    tables_for_player_attraction_recommendation,
    tables_for_map_result,
    tables_for_player_data_result,
)


def test_players_list_result_converts_to_table() -> None:
    result = PlayerDataResult(
        status=PlayerDataStatus.FOUND,
        summary="共查询到 2 条玩家数据。",
        data={
            "players": [
                {
                    "player_id": "1",
                    "nickname": "玩家一",
                    "level": 10,
                    "server_name": "一区",
                    "status": "active",
                    "desc": "进攻型玩家",
                },
                {
                    "player_id": "2",
                    "nickname": "玩家二",
                    "level": 20,
                    "server_name": "二区",
                    "status": "active",
                    "desc": "探索型玩家",
                },
            ]
        },
    )

    tables = tables_for_player_data_result(result, tool_name="mysql_players_list")

    assert len(tables) == 1
    table = tables[0]
    assert table.title == "玩家列表"
    assert [column.model_dump() for column in table.columns] == [
        {"key": "player_id", "label": "玩家ID"},
        {"key": "nickname", "label": "昵称"},
        {"key": "level", "label": "等级"},
        {"key": "server_name", "label": "服务器"},
        {"key": "status", "label": "状态"},
        {"key": "desc", "label": "个性描述"},
    ]
    assert table.rows[0]["nickname"] == "玩家一"
    assert table.rows[1]["desc"] == "探索型玩家"


def test_amap_place_search_result_converts_to_table() -> None:
    result = MapToolResult(
        status=MapToolStatus.FOUND,
        summary="高德地图查询结果。",
        data={
            "tool": "maps_text_search",
            "result": {
                "structuredContent": {
                    "pois": [
                        {
                            "name": "西湖风景名胜区",
                            "address": "杭州市西湖区龙井路1号",
                            "type": "风景名胜",
                            "distance": "2300",
                        }
                    ]
                }
            },
        },
    )

    tables = tables_for_map_result(result)

    assert len(tables) == 1
    table = tables[0]
    assert table.title == "高德地图地点结果"
    assert [column.model_dump() for column in table.columns] == [
        {"key": "name", "label": "名称"},
        {"key": "address", "label": "地址"},
        {"key": "type", "label": "类型"},
        {"key": "distance", "label": "距离"},
    ]
    assert table.rows == [
        {
            "name": "西湖风景名胜区",
            "address": "杭州市西湖区龙井路1号",
            "type": "风景名胜",
            "distance": "2300",
        }
    ]


def test_presentation_plan_uses_table_for_list_query_without_explicit_table() -> None:
    plan = build_presentation_plan("列出广州所有的旅游景点")

    assert plan.mode == "table"


def test_presentation_plan_respects_explicit_text_request() -> None:
    plan = build_presentation_plan("列出广州所有的旅游景点，不用表格，用文字说明")

    assert plan.mode == "text"


def test_presentation_plan_treats_desc_as_field_not_descending_order() -> None:
    plan = build_presentation_plan("查询所有玩家资料，根据desc进行分类，用表格显示")

    assert plan.sort_by == "desc"
    assert plan.sort_order == "asc"


def test_amap_place_search_result_sorts_by_distance_when_requested() -> None:
    result = MapToolResult(
        status=MapToolStatus.FOUND,
        summary="高德地图查询结果。",
        data={
            "tool": "maps_text_search",
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
                }
            },
        },
    )

    tables = tables_for_map_result(
        result,
        presentation_plan=PresentationPlan(mode="table", sort_by="distance", sort_order="asc"),
    )

    assert len(tables) == 1
    assert [row["name"] for row in tables[0].rows] == ["较近景点", "中等景点", "较远景点"]


def test_amap_place_search_result_skips_table_when_text_mode_requested() -> None:
    result = MapToolResult(
        status=MapToolStatus.FOUND,
        summary="高德地图查询结果。",
        data={
            "tool": "maps_text_search",
            "result": {
                "structuredContent": {
                    "pois": [
                        {
                            "name": "广州塔",
                            "address": "广州市海珠区阅江西路222号",
                            "type": "风景名胜",
                            "distance": "600",
                        }
                    ]
                }
            },
        },
    )

    tables = tables_for_map_result(result, presentation_plan=PresentationPlan(mode="text"))

    assert tables == []


def test_player_attraction_recommendation_combines_players_and_places() -> None:
    player_result = PlayerDataResult(
        status=PlayerDataStatus.FOUND,
        summary="共查询到 2 条玩家数据。",
        data={
            "players": [
                {
                    "player_id": "1",
                    "nickname": "玩家一",
                    "level": 10,
                    "server_name": "一区",
                    "status": "active",
                    "desc": "进攻型玩家",
                },
                {
                    "player_id": "2",
                    "nickname": "玩家二",
                    "level": 20,
                    "server_name": "二区",
                    "status": "active",
                    "desc": "探索型玩家",
                },
            ]
        },
    )
    map_result = MapToolResult(
        status=MapToolStatus.FOUND,
        summary="高德地图查询结果。",
        data={
            "tool": "maps_text_search",
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
                }
            },
        },
    )

    tables = tables_for_player_attraction_recommendation(player_result, map_result)

    assert len(tables) == 1
    table = tables[0]
    assert table.title == "玩家景点推荐"
    assert [column.model_dump() for column in table.columns] == [
        {"key": "player_id", "label": "玩家ID"},
        {"key": "nickname", "label": "昵称"},
        {"key": "desc", "label": "个性描述"},
        {"key": "recommended_place", "label": "推荐景点"},
        {"key": "recommendation_reason", "label": "推荐原因"},
        {"key": "expected_mood", "label": "游玩后可能收获"},
    ]
    assert table.rows[0]["recommended_place"] == "广州塔"
    assert "挑战" in table.rows[0]["expected_mood"]
    assert table.rows[1]["recommended_place"] == "广东省博物馆"
    assert "好奇心" in table.rows[1]["expected_mood"]
