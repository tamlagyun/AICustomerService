from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus
from app.table_adapter import tables_for_map_result, tables_for_player_data_result


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
