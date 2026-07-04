from __future__ import annotations

import logging
import re

from app.agent.decision import AgentAction, AgentDecision
from app.agent.planner import AgentPlan, PlanStep
from app.agent.state import CustomerServiceState
from app.config import Settings, get_settings
from app.knowledge_base import KnowledgeBaseSearch, KnowledgeChunk
from app.player_data import PlayerDataStatus
from app.tools.registry import is_map_tool_action

logger = logging.getLogger(__name__)


def correct_decision_with_high_confidence_rules(
    state: CustomerServiceState,
    decision: AgentDecision,
) -> AgentDecision:
    message = state.get("normalized_message", state.get("message", ""))
    overridable_actions = {
        AgentAction.ASK_CLARIFICATION,
        AgentAction.DIRECT_ANSWER,
        AgentAction.FALLBACK,
    }

    if (
        decision.action in overridable_actions
        and looks_like_player_attraction_batch_request(message)
    ):
        return players_attractions_decision(
            reason=f"{decision.reason}；后端高置信规则：需要先查询玩家列表再查询广州景点"
        )

    if decision.action in overridable_actions and looks_like_knowledge_issue(message):
        return AgentDecision(
            action=AgentAction.KNOWLEDGE_BASE,
            reason=f"{decision.reason}；后端高置信规则：客服知识类问题优先检索知识库",
        )

    return decision


def correct_plan_with_high_confidence_rules(
    state: CustomerServiceState,
    plan: AgentPlan,
) -> AgentPlan:
    message = state.get("normalized_message", state.get("message", ""))
    if not looks_like_player_attraction_batch_request(message):
        return plan

    if _plan_has_action(plan, AgentAction.MYSQL_PLAYERS_LIST) and _plan_has_map_action(plan):
        return plan

    return AgentPlan(
        final_task=players_attractions_final_task(),
        steps=[
            PlanStep(
                action=AgentAction.MYSQL_PLAYERS_LIST,
                reason="后端高置信规则：需要先查询玩家列表和 desc 字段",
                arguments={"limit": 100},
            ),
            PlanStep(
                action=AgentAction.AMAP_PLACE_SEARCH,
                reason="后端高置信规则：需要查询广州真实景点数据",
                arguments={
                    "keywords": "旅游景点",
                    "city": "广州",
                    "presentation": {"mode": "table"},
                },
                final_task=players_attractions_final_task(),
            ),
        ],
    )


def players_attractions_decision(reason: str) -> AgentDecision:
    return AgentDecision(
        action=AgentAction.MYSQL_PLAYERS_LIST,
        reason=reason,
        arguments={"limit": 100, "presentation": {"mode": "table"}},
        final_task=players_attractions_final_task(),
    )


def players_attractions_final_task() -> str:
    return "结合玩家资料、desc 类型和广州景点结果，推荐适合的景点并说明游玩后的心情收获"


def knowledge_results_for_overridable_decision(
    state: CustomerServiceState,
    decision: AgentDecision,
) -> list[KnowledgeChunk]:
    if state.get("knowledge_source") != "doc":
        return []
    if decision.action not in {
        AgentAction.ASK_CLARIFICATION,
        AgentAction.DIRECT_ANSWER,
        AgentAction.FALLBACK,
    }:
        return []

    message = state.get("normalized_message", state.get("message", ""))
    try:
        results = KnowledgeBaseSearch(
            get_settings().knowledge_base_dir,
            retrieval_mode="keyword",
            knowledge_source="doc",
        ).search(message, limit=1)
    except Exception:
        logger.exception("Knowledge precheck failed; session_id=%s", state.get("session_id"))
        return []

    if results and is_high_confidence_knowledge_match(message, results[0]):
        return results
    return []


def is_high_confidence_knowledge_match(message: str, chunk: KnowledgeChunk) -> bool:
    normalized_message = normalize_knowledge_match_text(message)
    normalized_title = normalize_knowledge_match_text(chunk.title)
    return bool(
        normalized_message
        and normalized_title
        and len(normalized_message) >= 2
        and (
            normalized_message == normalized_title
            or normalized_message in normalized_title
            or normalized_title in normalized_message
        )
    )


def normalize_knowledge_match_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def looks_like_players_list_query(message: str) -> bool:
    normalized = message.lower()
    asks_many = any(keyword in message for keyword in ["所有", "全部", "列表", "全量"])
    asks_player_rows = any(keyword in message for keyword in ["玩家", "资料", "数据", "信息"])

    if "players" in normalized and asks_many:
        return True
    if "数据库" in message and asks_many and asks_player_rows:
        return True
    return any(keyword in message for keyword in ["所有玩家", "全部玩家", "玩家列表"])


def looks_like_knowledge_issue(message: str) -> bool:
    return any(keyword in message for keyword in ["充值", "不到账", "订单", "封禁"])


def looks_like_player_attraction_batch_request(message: str) -> bool:
    has_city = "广州" in message
    asks_attractions = any(keyword in message for keyword in ["景点", "旅游", "游玩", "旅行", "玩"])
    mentions_player_profile = any(
        keyword in message
        for keyword in ["玩家资料", "玩家数据", "玩家特色", "desc", "个性", "类型", "他们"]
    )
    return has_city and asks_attractions and mentions_player_profile


def needs_attraction_recommendation(state: CustomerServiceState) -> bool:
    player_data_result = state.get("player_data_result")
    if player_data_result is None or player_data_result.status != PlayerDataStatus.FOUND:
        return False

    message = state.get("normalized_message", state.get("message", ""))
    has_city = "广州" in message
    asks_attractions = any(keyword in message for keyword in ["景点", "旅游", "游玩", "旅行", "玩"])
    asks_player_based_recommendation = any(
        keyword in message for keyword in ["玩家特色", "desc", "个性", "类型", "他们"]
    )
    return has_city and asks_attractions and asks_player_based_recommendation


def fallback_attraction_map_decision(state: CustomerServiceState) -> AgentDecision | None:
    if not needs_attraction_recommendation(state):
        return None
    return AgentDecision(
        action=AgentAction.AMAP_PLACE_SEARCH,
        reason="根据玩家资料推荐广州景点需要先查询真实景点数据",
        arguments={
            "keywords": "旅游景点",
            "city": "广州",
            "presentation": {"mode": "table"},
        },
        final_task=players_attractions_final_task(),
    )


def is_map_decision(decision: AgentDecision) -> bool:
    return is_map_tool_action(decision.action)


def normalize_knowledge_source(source: str | None, settings: Settings) -> str:
    selected_source = source or settings.knowledge_source_default
    normalized = selected_source.strip().lower()
    if normalized in {"doc", "vector"}:
        return normalized
    return "doc"


def _plan_has_action(plan: AgentPlan, action: AgentAction) -> bool:
    return any(step.action == action for step in plan.steps)


def _plan_has_map_action(plan: AgentPlan) -> bool:
    return any(is_map_decision(step.to_decision()) for step in plan.steps)
