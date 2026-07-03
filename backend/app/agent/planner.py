from dataclasses import dataclass
import json

from app.agent.decision import AgentAction, AgentDecision


class PlanParseError(ValueError):
    pass


@dataclass(frozen=True)
class PlanStep:
    action: AgentAction
    reason: str
    arguments: dict[str, object]
    final_task: str = ""
    direct_reply: str = ""

    def to_decision(self, *, fallback_final_task: str = "") -> AgentDecision:
        return AgentDecision(
            action=self.action,
            reason=self.reason,
            arguments=self.arguments,
            final_task=self.final_task or fallback_final_task,
            direct_reply=self.direct_reply,
        )


@dataclass(frozen=True)
class AgentPlan:
    steps: list[PlanStep]
    final_task: str = ""

    def actions(self) -> list[str]:
        return [str(step.action) for step in self.steps]


def parse_agent_plan(raw_content: str) -> AgentPlan:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise PlanParseError("Planner output is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise PlanParseError("Planner output must be a JSON object")

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanParseError("Planner output must include at least one step")

    steps = [_parse_step(raw_step) for raw_step in raw_steps]
    return AgentPlan(steps=steps, final_task=str(payload.get("final_task", "")))


def _parse_step(raw_step: object) -> PlanStep:
    if not isinstance(raw_step, dict):
        raise PlanParseError("Planner step must be a JSON object")

    raw_action = raw_step.get("action")
    try:
        action = AgentAction(raw_action)
    except ValueError as exc:
        raise PlanParseError(f"Unsupported planner action: {raw_action}") from exc

    return PlanStep(
        action=action,
        reason=str(raw_step.get("reason", "")),
        arguments=_parse_arguments(raw_step.get("arguments")),
        final_task=str(raw_step.get("final_task", "")),
        direct_reply=str(raw_step.get("direct_reply", "")),
    )


def _parse_arguments(raw_arguments: object) -> dict[str, object]:
    if isinstance(raw_arguments, dict):
        return {str(key): value for key, value in raw_arguments.items() if _is_jsonish(value)}
    return {}


def _is_jsonish(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool | list | dict)
