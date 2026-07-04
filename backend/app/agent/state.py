from __future__ import annotations

import asyncio
from typing import Literal, TypedDict

from app.agent.decision import AgentDecision
from app.agent.planner import AgentPlan
from app.agent.trace import AgentTrace
from app.avatar_generation import AvatarGenerationResult
from app.conversation_memory import ConversationMessage
from app.knowledge_base import KnowledgeChunk
from app.llm import LLMClientProtocol
from app.map_tools import MapToolResult
from app.player_data import PlayerDataResult
from app.safety import SafetyDecision
from app.schemas import ChatImage, ChatSource, ChatTable


QuestionType = Literal[
    "handoff",
    "knowledge",
    "general",
    "refuse",
    "player_data",
    "direct_answer",
    "map",
    "players_list",
]


class CustomerServiceState(TypedDict, total=False):
    session_id: str
    player_id: str | None
    message: str
    normalized_message: str
    knowledge_source: str
    question_type: QuestionType
    safety_decision: SafetyDecision
    llm_client: LLMClientProtocol | None
    llm_decision: AgentDecision
    map_decision: AgentDecision
    use_planner: bool
    agent_plan: AgentPlan
    agent_trace: AgentTrace
    plan_step_index: int
    completed_plan_steps: list[dict[str, object]]
    planner_fallback_reason: str
    use_llm_final_reply: bool
    knowledge_results: list[KnowledgeChunk]
    knowledge_precheck_results: list[KnowledgeChunk]
    knowledge_unavailable_reason: str
    player_data_result: PlayerDataResult
    map_result: MapToolResult
    avatar_result: AvatarGenerationResult
    conversation_history: list[ConversationMessage]
    reply: str
    sources: list[ChatSource]
    images: list[ChatImage]
    tables: list[ChatTable]
    handoff: bool
    status_queue: asyncio.Queue[str]
