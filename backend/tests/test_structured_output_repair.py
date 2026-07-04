from app.structured_output_repair import repair_structured_json


def test_repair_structured_json_extracts_markdown_json_block() -> None:
    result = repair_structured_json(
        """
        可以，计划如下：

        ```json
        {"action":"mysql_player_profile","arguments":{"player_id":"1"}}
        ```
        """
    )

    assert result.success is True
    assert result.repaired is True
    assert result.content == '{"action":"mysql_player_profile","arguments":{"player_id":"1"}}'


def test_repair_structured_json_extracts_object_from_explained_text() -> None:
    result = repair_structured_json(
        '我会调用工具：{"action":"amap_weather","arguments":{"city":"北京"}}，然后回答。'
    )

    assert result.success is True
    assert result.repaired is True
    assert result.content == '{"action":"amap_weather","arguments":{"city":"北京"}}'


def test_repair_structured_json_fixes_single_quotes_and_trailing_commas() -> None:
    result = repair_structured_json(
        "{'action':'mysql_players_list','arguments':{'limit':100,},}"
    )

    assert result.success is True
    assert result.repaired is True
    assert result.content == '{"action":"mysql_players_list","arguments":{"limit":100}}'


def test_repair_structured_json_wraps_missing_outer_object() -> None:
    result = repair_structured_json(
        '"action":"direct_answer","reason":"普通回复","direct_reply":"你好"'
    )

    assert result.success is True
    assert result.repaired is True
    assert result.content == (
        '{"action":"direct_answer","reason":"普通回复","direct_reply":"你好"}'
    )


def test_repair_structured_json_reports_clear_failure() -> None:
    result = repair_structured_json("我觉得不需要 JSON")

    assert result.success is False
    assert result.content == ""
    assert result.reason
