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


def test_parse_agent_decision_falls_back_on_invalid_json() -> None:
    decision = parse_agent_decision("我觉得应该查询玩家资料")

    assert decision.action == AgentAction.FALLBACK
    assert "无法解析" in decision.reason


def test_parse_agent_decision_rejects_unknown_action() -> None:
    decision = parse_agent_decision('{"action":"write_sql","reason":"直接写 SQL"}')

    assert decision.action == AgentAction.FALLBACK
    assert "不支持" in decision.reason
