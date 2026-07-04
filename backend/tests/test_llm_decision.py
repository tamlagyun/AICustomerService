from app.agent.decision import AgentAction, AgentDecision, parse_agent_decision


def test_parse_agent_decision_accepts_valid_json() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"mysql_player_profile","reason":"玩家要查询自己的资料",'
            '"arguments":{"player_id":"1"},"final_task":"分析总结玩家个性","direct_reply":""}'
        )
    )

    assert decision == AgentDecision(
        action=AgentAction.MYSQL_PLAYER_PROFILE,
        reason="玩家要查询自己的资料",
        arguments={"player_id": "1"},
        final_task="分析总结玩家个性",
        direct_reply="",
    )


def test_parse_agent_decision_accepts_clarification_action() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"ask_clarification","reason":"缺少玩家 ID",'
            '"arguments":{"missing":["player_id"]},'
            '"direct_reply":"请提供玩家 ID，我才能查询资料。"}'
        )
    )

    assert decision.action == AgentAction.ASK_CLARIFICATION
    assert decision.arguments == {"missing": ["player_id"]}
    assert decision.direct_reply == "请提供玩家 ID，我才能查询资料。"


def test_parse_agent_decision_accepts_players_list_action() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"mysql_players_list","reason":"用户要查询所有玩家",'
            '"arguments":{"limit":1000},"final_task":"总结玩家列表整体情况"}'
        )
    )

    assert decision.action == AgentAction.MYSQL_PLAYERS_LIST
    assert decision.arguments == {"limit": 1000}
    assert decision.final_task == "总结玩家列表整体情况"


def test_parse_agent_decision_defaults_players_list_limit() -> None:
    decision = parse_agent_decision(
        '{"action":"mysql_players_list","reason":"用户要查询所有玩家"}'
    )

    assert decision.action == AgentAction.MYSQL_PLAYERS_LIST
    assert decision.arguments == {"limit": 100}


def test_parse_agent_decision_clamps_players_list_limit() -> None:
    decision = parse_agent_decision(
        '{"action":"mysql_players_list","reason":"用户要查询所有玩家","arguments":{"limit":"2000"}}'
    )

    assert decision.action == AgentAction.MYSQL_PLAYERS_LIST
    assert decision.arguments == {"limit": 1000}


def test_parse_agent_decision_rejects_invalid_tool_arguments() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"amap_navigation","reason":"玩家要求打开导航",'
            '"arguments":{"destination":"天安门","mode":"spaceship"}}'
        )
    )

    assert decision.action == AgentAction.FALLBACK
    assert "工具参数不合法" in decision.reason


def test_parse_agent_decision_accepts_avatar_generate_action() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"avatar_generate","reason":"玩家要求根据资料生成头像",'
            '"arguments":{"player_id":"1"},"final_task":"生成符合玩家个性的头像"}'
        )
    )

    assert decision.action == AgentAction.AVATAR_GENERATE
    assert decision.arguments == {"player_id": "1"}
    assert decision.final_task == "生成符合玩家个性的头像"


def test_parse_agent_decision_accepts_amap_place_search_action() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"amap_place_search","reason":"玩家询问附近地点",'
            '"arguments":{"keywords":"网吧","city":"北京"},'
            '"final_task":"回答玩家可选择的附近地点"}'
        )
    )

    assert decision.action == AgentAction.AMAP_PLACE_SEARCH
    assert decision.arguments == {"keywords": "网吧", "city": "北京"}
    assert decision.final_task == "回答玩家可选择的附近地点"


def test_parse_agent_decision_accepts_amap_navigation_action() -> None:
    decision = parse_agent_decision(
        (
            '{"action":"amap_navigation","reason":"玩家要求打开导航",'
            '"arguments":{"destination":"天安门","city":"北京","mode":"walking"},'
            '"final_task":"提供高德地图导航链接"}'
        )
    )

    assert decision.action == AgentAction.AMAP_NAVIGATION
    assert decision.arguments == {"destination": "天安门", "city": "北京", "mode": "walking"}
    assert decision.final_task == "提供高德地图导航链接"


def test_parse_agent_decision_accepts_amap_weather_action() -> None:
    decision = parse_agent_decision(
        '{"action":"amap_weather","reason":"玩家询问天气","arguments":{"city":"北京"}}'
    )

    assert decision.action == AgentAction.AMAP_WEATHER
    assert decision.arguments == {"city": "北京"}


def test_parse_agent_decision_falls_back_on_invalid_json() -> None:
    decision = parse_agent_decision("我觉得应该查询玩家资料")

    assert decision.action == AgentAction.FALLBACK
    assert "无法解析" in decision.reason


def test_parse_agent_decision_rejects_unknown_action() -> None:
    decision = parse_agent_decision('{"action":"write_sql","reason":"直接写 SQL"}')

    assert decision.action == AgentAction.FALLBACK
    assert "不支持" in decision.reason
