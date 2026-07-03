import pytest

from app.agent.decision import AgentAction
from app.agent.planner import PlanParseError, parse_agent_plan


def test_parse_agent_plan_reads_ordered_steps() -> None:
    plan = parse_agent_plan(
        """
        {
          "final_task": "combine player profile and attractions",
          "steps": [
            {
              "action": "mysql_player_profile",
              "reason": "need player profile",
              "arguments": {"player_id": "1"}
            },
            {
              "action": "amap_place_search",
              "reason": "need attractions",
              "arguments": {"keywords": "attractions", "city": "Guangzhou"}
            }
          ]
        }
        """
    )

    assert plan.final_task == "combine player profile and attractions"
    assert [step.action for step in plan.steps] == [
        AgentAction.MYSQL_PLAYER_PROFILE,
        AgentAction.AMAP_PLACE_SEARCH,
    ]
    assert plan.steps[0].arguments == {"player_id": "1"}
    assert plan.steps[1].arguments == {"keywords": "attractions", "city": "Guangzhou"}


def test_parse_agent_plan_rejects_unknown_action() -> None:
    with pytest.raises(PlanParseError, match="Unsupported planner action"):
        parse_agent_plan('{"steps":[{"action":"write_sql","reason":"bad"}]}')


def test_parse_agent_plan_rejects_empty_steps() -> None:
    with pytest.raises(PlanParseError, match="at least one step"):
        parse_agent_plan('{"steps":[]}')
