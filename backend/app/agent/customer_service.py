import asyncio
from functools import lru_cache
import logging
from typing import Any, AsyncIterator, Literal

from langgraph.graph import END, StateGraph

from app.agent.audit import write_chat_audit_event
from app.agent.checkpoint import write_chat_checkpoint_event
from app.agent.context_budget import ContextBudget, apply_context_budget
from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import MapAgentResult, run_map_agent
from app.agent.planner import PlanParseError, parse_agent_plan
from app.agent.policies import (
    correct_decision_with_high_confidence_rules,
    correct_plan_with_high_confidence_rules,
    fallback_attraction_map_decision,
    is_map_decision,
    knowledge_results_for_overridable_decision,
    looks_like_knowledge_issue,
    looks_like_players_list_query,
    needs_attraction_recommendation,
    normalize_knowledge_source,
)
from app.agent.prompting import (
    decision_messages,
    final_reply_messages,
    followup_decision_messages,
    log_prompt_versions,
    map_tool_name,
    planner_messages,
    should_use_llm_final_reply,
    tool_name_for_prompt,
)
from app.agent.state import CustomerServiceState, QuestionType
from app.agent.streaming import StreamingLLMClient
from app.agent.trace import AgentTrace, TraceEventType
from app.avatar_generation import build_avatar_generator
from app.config import Settings, get_settings
from app.conversation_memory import get_conversation_memory
from app.knowledge_base import KnowledgeBaseSearch
from app.llm import LLMClientProtocol, build_llm_client
from app.llm_usage import LLMUsageSummary
from app.llm_middleware import LLMCallContext, wrap_llm_client
from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus, build_player_data_tools
from app.prompt_registry import PromptNotFoundError
from app.rag.chroma_store import ChromaIndexNotReady, ChromaUnavailableError, EmbeddingProviderError
from app.safety import SafetyAction, analyze_safety, redact_sensitive_text
from app.schemas import ChatImage, ChatResponse, ChatSource
from app.table_adapter import (
    PresentationPlan,
    build_presentation_plan,
    tables_for_player_attraction_recommendation,
    tables_for_map_result,
    tables_for_player_data_result,
)
from app.tools.executor import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolExecutionStatus,
    execute_tool_action,
)
from app.tools.registry import ToolCategory, get_tool_by_action

logger = logging.getLogger(__name__)


def build_customer_service_graph():
    workflow = StateGraph(CustomerServiceState)
    workflow.add_node("analyze_safety", analyze_safety_node)
    workflow.add_node("plan_with_llm", plan_with_llm)
    workflow.add_node("decide_action_with_llm", decide_action_with_llm)
    workflow.add_node("classify_question", classify_question)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("retrieve_player_data", retrieve_player_data)
    workflow.add_node("retrieve_players_list", retrieve_players_list)
    workflow.add_node("decide_followup_after_player_data", decide_followup_after_player_data)
    workflow.add_node("retrieve_map_data", retrieve_map_data)
    workflow.add_node("generate_avatar", generate_avatar)
    workflow.add_node("generate_refusal_reply", generate_refusal_reply)
    workflow.add_node("generate_handoff_reply", generate_handoff_reply)
    workflow.add_node("generate_general_reply", generate_general_reply)
    workflow.add_node("generate_knowledge_reply", generate_knowledge_reply)
    workflow.add_node("generate_player_data_reply", generate_player_data_reply)
    workflow.add_node("generate_map_reply", generate_map_reply)
    workflow.add_node("generate_avatar_reply", generate_avatar_reply)
    workflow.add_node("generate_direct_reply", generate_direct_reply)
    workflow.add_node("generate_no_knowledge_reply", generate_no_knowledge_reply)
    workflow.add_node("generate_llm_final_reply", generate_llm_final_reply)
    workflow.add_node("finalize", finalize_response)

    workflow.set_entry_point("analyze_safety")
    workflow.add_conditional_edges(
        "analyze_safety",
        route_safety,
        {
            "refuse": "generate_refusal_reply",
            "handoff": "generate_handoff_reply",
            "allow": "plan_with_llm",
        },
    )
    workflow.add_conditional_edges(
        "plan_with_llm",
        route_after_planning,
        {
            "legacy": "decide_action_with_llm",
            "fallback": "decide_action_with_llm",
            "handoff": "generate_handoff_reply",
            "direct_answer": "generate_direct_reply",
            "ask_clarification": "generate_direct_reply",
            "knowledge": "retrieve_knowledge",
            "player_data": "retrieve_player_data",
            "players_list": "retrieve_players_list",
            "avatar": "retrieve_player_data",
            "map": "retrieve_map_data",
        },
    )
    workflow.add_conditional_edges(
        "decide_action_with_llm",
        route_llm_decision,
        {
            "fallback": "classify_question",
            "handoff": "generate_handoff_reply",
            "direct_answer": "generate_direct_reply",
            "ask_clarification": "generate_direct_reply",
            "knowledge": "retrieve_knowledge",
            "player_data": "retrieve_player_data",
            "players_list": "retrieve_players_list",
            "avatar": "retrieve_player_data",
            "map": "retrieve_map_data",
        },
    )
    workflow.add_conditional_edges(
        "classify_question",
        route_question,
        {
            "handoff": "generate_handoff_reply",
            "player_data": "retrieve_player_data",
            "players_list": "retrieve_players_list",
            "knowledge": "retrieve_knowledge",
            "general": "retrieve_knowledge",
            "map": "retrieve_map_data",
        },
    )
    workflow.add_conditional_edges(
        "retrieve_knowledge",
        route_after_knowledge,
        {
            "knowledge": "generate_knowledge_reply",
            "general": "generate_general_reply",
            "fallback": "generate_no_knowledge_reply",
        },
    )
    workflow.add_conditional_edges(
        "retrieve_player_data",
        route_after_player_data,
        {
            "avatar": "generate_avatar",
            "player_data": "decide_followup_after_player_data",
        },
    )
    workflow.add_edge("retrieve_players_list", "decide_followup_after_player_data")
    workflow.add_conditional_edges(
        "decide_followup_after_player_data",
        route_after_followup_decision,
        {
            "map": "retrieve_map_data",
            "player_data": "generate_player_data_reply",
        },
    )
    workflow.add_edge("retrieve_map_data", "generate_map_reply")
    workflow.add_edge("generate_avatar", "generate_avatar_reply")
    workflow.add_edge("generate_refusal_reply", "finalize")
    workflow.add_edge("generate_handoff_reply", "finalize")
    workflow.add_conditional_edges(
        "generate_general_reply",
        route_final_reply,
        {"llm": "generate_llm_final_reply", "final": "finalize"},
    )
    workflow.add_conditional_edges(
        "generate_knowledge_reply",
        route_final_reply,
        {"llm": "generate_llm_final_reply", "final": "finalize"},
    )
    workflow.add_conditional_edges(
        "generate_player_data_reply",
        route_final_reply,
        {"llm": "generate_llm_final_reply", "final": "finalize"},
    )
    workflow.add_conditional_edges(
        "generate_avatar_reply",
        route_final_reply,
        {"llm": "generate_llm_final_reply", "final": "finalize"},
    )
    workflow.add_conditional_edges(
        "generate_map_reply",
        route_final_reply,
        {"llm": "generate_llm_final_reply", "final": "finalize"},
    )
    workflow.add_edge("generate_direct_reply", "finalize")
    workflow.add_edge("generate_no_knowledge_reply", "finalize")
    workflow.add_edge("generate_llm_final_reply", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()


async def run_customer_service_agent(
    *,
    session_id: str,
    message: str,
    player_id: str | None = None,
    model_provider: str | None = None,
    use_planner: bool = False,
    knowledge_source: str | None = None,
    llm_client: LLMClientProtocol | None = None,
    status_queue: asyncio.Queue[str] | None = None,
) -> ChatResponse:
    settings = get_settings()
    memory = get_conversation_memory()
    capability_reply = _system_capability_reply(message)
    if capability_reply is not None:
        response = ChatResponse(reply=capability_reply)
        _record_conversation_exchange(session_id, message, response.reply)
        write_chat_audit_event(
            settings,
            session_id=session_id,
            player_id=player_id,
            message=message,
            response=response,
        )
        _write_chat_checkpoint_safely(
            settings,
            session_id=session_id,
            player_id=player_id,
            message=message,
            response=response,
        )
        return response

    agent_trace = AgentTrace()
    llm_usage_summary = LLMUsageSummary()
    base_llm_client = llm_client if llm_client is not None else build_llm_client(model_provider)
    selected_llm_client = _wrap_request_llm_client(
        base_llm_client,
        session_id=session_id,
        provider_name=_llm_context_provider(
            settings,
            requested_provider=model_provider,
            injected=llm_client is not None,
        ),
        agent_trace=agent_trace,
        usage_summary=llm_usage_summary,
        settings=settings,
    )
    if selected_llm_client is not None:
        log_prompt_versions(session_id, settings)

    final_state = await _compiled_graph().ainvoke(
        {
            "session_id": session_id,
            "player_id": player_id,
            "message": message,
            "use_planner": use_planner,
            "knowledge_source": normalize_knowledge_source(knowledge_source, settings),
            "conversation_history": memory.get_recent_messages(session_id),
            "llm_client": selected_llm_client,
            "llm_usage_summary": llm_usage_summary,
            "status_queue": status_queue,
            "agent_trace": agent_trace,
        }
    )
    response = ChatResponse(
        reply=redact_sensitive_text(final_state["reply"]),
        sources=final_state.get("sources", []),
        handoff=final_state.get("handoff", False),
        images=final_state.get("images", []),
        tables=final_state.get("tables", []),
    )
    _record_conversation_exchange(session_id, message, response.reply)
    write_chat_audit_event(
        settings,
        session_id=session_id,
        player_id=player_id,
        message=message,
        response=response,
        final_state=final_state,
    )
    _write_chat_checkpoint_safely(
        settings,
        session_id=session_id,
        player_id=player_id,
        message=message,
        response=response,
        final_state=final_state,
    )
    return response


def _system_capability_reply(message: str) -> str | None:
    normalized = message.strip().lower()
    asks_streaming = any(keyword in normalized for keyword in ["流式", "sse", "stream"])
    asks_chat_transport = any(keyword in normalized for keyword in ["输出", "返回", "接口", "模式", "采用"])
    if asks_streaming and asks_chat_transport:
        return (
            "当前聊天已采用 SSE 流式输出。前端默认调用 `/api/chat/stream`，"
            "后端通过 `text/event-stream` 返回 `status`、`token` 和 `done` 事件；"
            "旧的 `/api/chat` 普通 REST 一次性返回接口仍保留用于兼容。"
        )

    return None


async def stream_customer_service_agent(
    *,
    session_id: str,
    message: str,
    player_id: str | None = None,
    model_provider: str | None = None,
    use_planner: bool = False,
    knowledge_source: str | None = None,
    llm_client: LLMClientProtocol | None = None,
) -> AsyncIterator[dict[str, Any]]:
    yield {"event": "status", "data": {"message": "正在分析问题"}}

    base_llm_client = llm_client if llm_client is not None else build_llm_client(model_provider)
    token_queue: asyncio.Queue[str] = asyncio.Queue()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    streaming_llm_client = (
        StreamingLLMClient(base_llm_client, token_queue) if base_llm_client is not None else None
    )

    task = asyncio.create_task(
        run_customer_service_agent(
            session_id=session_id,
            player_id=player_id,
            message=message,
            model_provider=model_provider,
            use_planner=use_planner,
            knowledge_source=knowledge_source,
            llm_client=streaming_llm_client,
            status_queue=status_queue,
        )
    )

    streamed_text = ""
    while not task.done() or not token_queue.empty() or not status_queue.empty():
        emitted = False
        while not status_queue.empty():
            emitted = True
            yield {"event": "status", "data": {"message": status_queue.get_nowait()}}
        while not token_queue.empty():
            emitted = True
            token = token_queue.get_nowait()
            streamed_text += token
            yield {"event": "token", "data": {"text": token}}
        if not emitted and not task.done():
            await asyncio.sleep(0.05)

    response = await task
    if not streamed_text:
        yield {"event": "token", "data": {"text": response.reply}}

    yield {
        "event": "done",
        "data": {
            "sources": [source.model_dump() for source in response.sources],
            "handoff": response.handoff,
            "images": [image.model_dump() for image in response.images],
            "tables": [table.model_dump() for table in response.tables],
        },
    }


def analyze_safety_node(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在检查安全策略")
    started_at = _trace_started(state, TraceEventType.NODE_STARTED, "analyze_safety")
    try:
        normalized_message = state["message"].strip()
        next_state = {
            **state,
            "normalized_message": normalized_message,
            "safety_decision": analyze_safety(normalized_message),
        }
    except Exception as exc:
        _trace_finished(state, TraceEventType.NODE_FINISHED, "analyze_safety", started_at, error=str(exc))
        raise
    _trace_finished(next_state, TraceEventType.NODE_FINISHED, "analyze_safety", started_at)
    return next_state


def route_safety(state: CustomerServiceState) -> Literal["allow", "handoff", "refuse"]:
    action = state["safety_decision"].action
    if action == SafetyAction.REFUSE:
        return "refuse"
    if action == SafetyAction.HANDOFF:
        return "handoff"
    return "allow"


async def plan_with_llm(state: CustomerServiceState) -> CustomerServiceState:
    if not state.get("use_planner"):
        return state

    started_at = _trace_started(state, TraceEventType.PLANNER_STARTED, "plan_with_llm")
    llm_client = state.get("llm_client")
    if llm_client is None:
        _trace_finished(
            state,
            TraceEventType.PLANNER_FINISHED,
            "plan_with_llm",
            started_at,
            {"fallback_reason": "llm_unavailable"},
        )
        return {**state, "planner_fallback_reason": "llm_unavailable"}

    try:
        _emit_status(state, "正在请求纯模型 Planner 生成执行计划")
        response = await llm_client.generate_reply(
            planner_messages(
                state["normalized_message"],
                state.get("conversation_history", []),
            )
        )
        plan = correct_plan_with_high_confidence_rules(
            state,
            parse_agent_plan(response.content),
        )
    except PromptNotFoundError:
        _trace_finished(
            state,
            TraceEventType.PLANNER_FINISHED,
            "plan_with_llm",
            started_at,
            error="PromptNotFoundError",
        )
        logger.exception("Prompt loading failed during Planner; session_id=%s", state.get("session_id"))
        raise
    except PlanParseError as exc:
        _trace_finished(
            state,
            TraceEventType.PLANNER_FINISHED,
            "plan_with_llm",
            started_at,
            error=type(exc).__name__,
        )
        logger.exception("Planner failed; session_id=%s", state.get("session_id"))
        _emit_status(state, f"Planner 生成计划失败，正在回退旧决策流程：{type(exc).__name__}")
        return {**state, "planner_fallback_reason": type(exc).__name__}
    except Exception as exc:
        _trace_finished(
            state,
            TraceEventType.PLANNER_FINISHED,
            "plan_with_llm",
            started_at,
            error=type(exc).__name__,
        )
        logger.exception("Planner failed; session_id=%s", state.get("session_id"))
        _emit_status(state, f"Planner 生成计划失败，正在回退旧决策流程：{type(exc).__name__}")
        return {**state, "planner_fallback_reason": type(exc).__name__}

    first_decision = plan.steps[0].to_decision(fallback_final_task=plan.final_task)
    next_state = {
        **state,
        "agent_plan": plan,
        "plan_step_index": 0,
        "completed_plan_steps": [],
        "llm_decision": first_decision,
    }
    _trace_finished(
        next_state,
        TraceEventType.PLANNER_FINISHED,
        "plan_with_llm",
        started_at,
        {"steps": len(plan.steps), "actions": plan.actions()},
    )
    return next_state


def route_after_planning(
    state: CustomerServiceState,
) -> Literal[
    "legacy",
    "fallback",
    "handoff",
    "direct_answer",
    "ask_clarification",
    "knowledge",
    "player_data",
    "players_list",
    "avatar",
    "map",
]:
    if not state.get("use_planner") or state.get("agent_plan") is None:
        return "legacy"
    return route_llm_decision(state)


async def decide_action_with_llm(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在判断是否需要调用工具")
    llm_client = state.get("llm_client")
    if llm_client is None:
        _emit_status(state, "正在使用本地规则决策")
        return state

    started_at = _trace_started(state, TraceEventType.LLM_STARTED, "decide_action")
    try:
        _emit_status(state, "正在请求大模型决策")
        messages = _apply_context_budget_to_messages(
            state,
            decision_messages(
                state["normalized_message"],
                state.get("conversation_history", []),
            ),
            protected_text=state["normalized_message"],
        )
        decision = await llm_client.decide_action(
            messages
        )
    except PromptNotFoundError:
        _trace_finished(
            state,
            TraceEventType.LLM_FINISHED,
            "decide_action",
            started_at,
            error="PromptNotFoundError",
        )
        logger.exception("Prompt loading failed during LLM decision; session_id=%s", state.get("session_id"))
        raise
    except Exception as exc:
        _trace_finished(
            state,
            TraceEventType.LLM_FINISHED,
            "decide_action",
            started_at,
            error=type(exc).__name__,
        )
        logger.exception("LLM decision failed; session_id=%s", state.get("session_id"))
        _emit_status(state, f"大模型决策失败，正在使用本地规则决策：{type(exc).__name__}")
        return state

    corrected_decision = correct_decision_with_high_confidence_rules(state, decision)
    next_state = {**state, "llm_decision": corrected_decision}
    _trace_finished(
        next_state,
        TraceEventType.LLM_FINISHED,
        "decide_action",
        started_at,
        {"action": str(corrected_decision.action)},
    )
    knowledge_results = knowledge_results_for_overridable_decision(next_state, corrected_decision)
    if knowledge_results:
        return {
            **next_state,
            "knowledge_precheck_results": knowledge_results,
            "llm_decision": AgentDecision(
                action=AgentAction.KNOWLEDGE_BASE,
                reason=(
                    f"{corrected_decision.reason}；后端高置信规则：当前问题精确命中知识库标题"
                ),
            ),
        }
    return next_state


def route_llm_decision(
    state: CustomerServiceState,
) -> Literal[
    "fallback",
    "handoff",
    "direct_answer",
    "ask_clarification",
    "knowledge",
    "player_data",
    "players_list",
    "avatar",
    "map",
]:
    decision = state.get("llm_decision")
    if decision is None or decision.action == AgentAction.FALLBACK:
        return "fallback"
    if decision.action == AgentAction.HANDOFF:
        return "handoff"
    if decision.action == AgentAction.DIRECT_ANSWER:
        return "direct_answer"
    if decision.action == AgentAction.ASK_CLARIFICATION:
        return "ask_clarification"

    tool = get_tool_by_action(decision.action)
    if tool is None:
        return "fallback"
    if tool.category == ToolCategory.KNOWLEDGE:
        return "knowledge"
    if tool.name == "mysql_player_profile":
        return "player_data"
    if tool.name == "mysql_players_list":
        return "players_list"
    if tool.category == ToolCategory.AVATAR:
        return "avatar"
    if tool.category == ToolCategory.MAP:
        return "map"
    return "fallback"


def classify_question(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在使用本地规则决策")
    normalized_message = state["normalized_message"]
    question_type: QuestionType = "general"

    if looks_like_players_list_query(normalized_message):
        question_type = "players_list"
    elif any(keyword in normalized_message for keyword in ["玩家资料", "玩家信息", "角色资料", "角色信息"]):
        question_type = "player_data"
    elif state.get("player_id") and any(keyword in normalized_message for keyword in ["资料", "信息"]):
        question_type = "player_data"
    elif looks_like_knowledge_issue(normalized_message):
        question_type = "knowledge"
    elif any(
        keyword in normalized_message
        for keyword in ["地图", "地址", "在哪里", "附近", "怎么去", "路线", "距离", "导航", "天气"]
    ):
        question_type = "map"

    return {
        **state,
        "normalized_message": normalized_message,
        "question_type": question_type,
    }


def route_question(state: CustomerServiceState) -> QuestionType:
    return state.get("question_type", "general")


def route_after_knowledge(state: CustomerServiceState) -> Literal["knowledge", "general", "fallback"]:
    if state.get("knowledge_unavailable_reason"):
        return "fallback"
    if state.get("knowledge_results"):
        return "knowledge"
    if state.get("question_type") == "general":
        return "general"
    return "fallback"


def route_after_player_data(state: CustomerServiceState) -> Literal["avatar", "player_data"]:
    decision = state.get("llm_decision")
    if decision is not None and decision.action == AgentAction.AVATAR_GENERATE:
        return "avatar"
    return "player_data"


def retrieve_knowledge(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在检索知识库")
    started_at = _trace_started(state, TraceEventType.NODE_STARTED, "retrieve_knowledge")
    precheck_results = state.get("knowledge_precheck_results")
    try:
        if precheck_results is not None:
            results = precheck_results
        else:
            results = KnowledgeBaseSearch(
                get_settings().knowledge_base_dir,
                knowledge_source=state.get("knowledge_source"),
            ).search(
                state["normalized_message"],
                limit=1,
            )
        knowledge_unavailable_reason = ""
    except (ChromaIndexNotReady, ChromaUnavailableError, EmbeddingProviderError) as exc:
        logger.exception("Knowledge vector search failed; session_id=%s", state.get("session_id"))
        results = []
        knowledge_unavailable_reason = str(exc)
        _emit_status(state, "向量知识库不可用，正在准备提示")
    next_state = {
        **state,
        "knowledge_results": results,
        "knowledge_unavailable_reason": knowledge_unavailable_reason,
        "use_llm_final_reply": bool(results)
        and precheck_results is None
        and should_use_llm_final_reply(state),
    }
    _trace_finished(
        next_state,
        TraceEventType.NODE_FINISHED,
        "retrieve_knowledge",
        started_at,
        {"result_count": len(results)},
        error=knowledge_unavailable_reason,
    )
    return _complete_current_plan_step(next_state)


async def retrieve_player_data(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在查询 MySQL 玩家数据")
    started_at = _trace_started(state, TraceEventType.NODE_STARTED, "retrieve_player_data")
    execution = await execute_tool_action(
        _player_profile_decision_for_tool_call(state),
        _tool_execution_context(state),
    )
    result = _player_data_result_from_execution(execution)
    next_state = {
        **state,
        "player_data_result": result,
        "use_llm_final_reply": should_use_llm_final_reply(state),
    }
    _trace_finished(
        next_state,
        TraceEventType.NODE_FINISHED,
        "retrieve_player_data",
        started_at,
        {"status": str(result.status)},
        error=execution.error,
    )
    current_plan_decision = _current_plan_decision(state)
    if current_plan_decision is not None and current_plan_decision.action == AgentAction.AVATAR_GENERATE:
        return next_state
    return _complete_current_plan_step(next_state)


async def retrieve_players_list(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在查询 MySQL 玩家列表")
    started_at = _trace_started(state, TraceEventType.NODE_STARTED, "retrieve_players_list")
    execution = await execute_tool_action(
        _players_list_decision_for_tool_call(state),
        _tool_execution_context(state),
    )
    result = _player_data_result_from_execution(execution)
    next_state = {
        **state,
        "player_data_result": result,
        "use_llm_final_reply": should_use_llm_final_reply(state),
    }
    _trace_finished(
        next_state,
        TraceEventType.NODE_FINISHED,
        "retrieve_players_list",
        started_at,
        {"status": str(result.status)},
        error=execution.error,
    )
    return _complete_current_plan_step(next_state)


async def decide_followup_after_player_data(state: CustomerServiceState) -> CustomerServiceState:
    planned_decision = _current_plan_decision(state)
    if planned_decision is not None and is_map_decision(planned_decision):
        return {**state, "map_decision": planned_decision}

    if not needs_attraction_recommendation(state):
        return state

    _emit_status(state, "正在根据玩家资料判断是否需要继续调用地图工具")
    llm_client = state.get("llm_client")
    if llm_client is not None:
        try:
            decision = await llm_client.decide_action(followup_decision_messages(state))
            if is_map_decision(decision):
                return {**state, "map_decision": decision}
        except PromptNotFoundError:
            logger.exception(
                "Prompt loading failed during LLM followup decision; session_id=%s",
                state.get("session_id"),
            )
            raise
        except Exception as exc:
            logger.exception("LLM followup decision failed; session_id=%s", state.get("session_id"))
            _emit_status(state, f"大模型后续决策失败，正在使用本地规则继续：{type(exc).__name__}")

    fallback_decision = fallback_attraction_map_decision(state)
    if fallback_decision is None:
        return state
    return {**state, "map_decision": fallback_decision}


def route_after_followup_decision(state: CustomerServiceState) -> Literal["map", "player_data"]:
    decision = state.get("map_decision")
    if decision is not None and is_map_decision(decision):
        return "map"
    return "player_data"


async def retrieve_map_data(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在委托地图 Agent")
    started_at = _trace_started(state, TraceEventType.NODE_STARTED, "retrieve_map_data")
    decision = _map_decision_for_tool_call(state)
    execution = await execute_tool_action(
        decision,
        _tool_execution_context(state),
    )
    map_agent_result = _map_agent_result_from_execution(execution, decision)

    next_state = {
        **state,
        "map_decision": map_agent_result.decision,
        "map_result": map_agent_result.map_result,
        "use_llm_final_reply": should_use_llm_final_reply(state),
    }
    _trace_finished(
        next_state,
        TraceEventType.NODE_FINISHED,
        "retrieve_map_data",
        started_at,
        {"status": str(map_agent_result.map_result.status)},
        error=execution.error,
    )
    return _complete_current_plan_step(next_state)


def generate_refusal_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在准备拒答回复")
    return {
        **state,
        "reply": state["safety_decision"].reply,
        "sources": [],
        "handoff": False,
    }


def generate_handoff_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在准备转人工回复")
    return {
        **state,
        "reply": state["safety_decision"].reply,
        "sources": [],
        "handoff": True,
    }


def generate_general_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在准备回复")
    normalized_message = state.get("normalized_message", "").strip()
    if not normalized_message:
        reply = "请描述你遇到的问题，我会尽力协助。"
    else:
        player_hint = f"玩家 {state['player_id']}，" if state.get("player_id") else ""
        reply = (
            f"{player_hint}我已收到你的问题：{normalized_message}。"
            "当前是基础客服 Agent，后续会接入 MySQL 玩家数据和知识库检索。"
        )

    return {
        **state,
        "reply": reply,
        "sources": [],
        "handoff": False,
    }


def generate_direct_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在准备回复")
    decision = state["llm_decision"]
    return {
        **state,
        "reply": decision.direct_reply or "请描述你遇到的问题，我会尽力协助。",
        "sources": [],
        "handoff": False,
    }


def generate_player_data_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在整理工具结果")
    player_data_result = state["player_data_result"]
    presentation_plan = _presentation_plan_for_state(state)
    return {
        **state,
        "reply": player_data_result.summary,
        "sources": [],
        "tables": tables_for_player_data_result(
            player_data_result,
            tool_name=tool_name_for_prompt(state),
            presentation_plan=presentation_plan,
        ),
        "handoff": False,
    }


def generate_map_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在整理地图结果")
    map_result = state["map_result"]
    presentation_plan = _presentation_plan_for_state(state)
    player_data_result = state.get("player_data_result")
    sources = []
    tool_name = map_tool_name(map_result)
    if tool_name:
        sources.append(
            ChatSource(
                title="高德地图 MCP",
                source_type="amap_mcp",
                reference=tool_name,
            )
        )

    tables = (
        tables_for_player_attraction_recommendation(
            player_data_result,
            map_result,
            presentation_plan=presentation_plan,
        )
        if player_data_result is not None
        else tables_for_map_result(map_result, presentation_plan=presentation_plan)
    )

    return {
        **state,
        "reply": _combined_player_map_summary(state, map_result.summary),
        "sources": sources,
        "tables": tables,
        "handoff": False,
    }


def generate_avatar(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在根据玩家资料生成头像")
    player_data_result = state.get("player_data_result")
    if (
        player_data_result is None
        or player_data_result.status != PlayerDataStatus.FOUND
        or not player_data_result.data
    ):
        reply = (
            player_data_result.summary
            if player_data_result is not None
            else "需要先查询到玩家资料，才能生成个性头像。"
        )
        return _complete_current_plan_step(
            {
                **state,
                "reply": reply,
                "sources": [],
                "images": [],
                "handoff": False,
            }
        )

    avatar_result = build_avatar_generator().generate_player_avatar(
        player_data_result.data,
        session_id=state["session_id"],
    )
    images = (
        [ChatImage(url=avatar_result.url, alt=avatar_result.alt)]
        if avatar_result.url and avatar_result.alt
        else []
    )
    return _complete_current_plan_step(
        {
            **state,
            "avatar_result": avatar_result,
            "reply": avatar_result.summary,
            "sources": [],
            "images": images,
            "handoff": False,
        }
    )


def generate_avatar_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在整理头像结果")
    return {
        **state,
        "sources": state.get("sources", []),
        "images": state.get("images", []),
        "handoff": False,
    }


def generate_no_knowledge_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在准备回复")
    unavailable_reason = state.get("knowledge_unavailable_reason")
    if unavailable_reason:
        reply = (
            f"{unavailable_reason}。请先在 Agent 评测页面点击“重建知识库向量库”；"
            "如果刚重建过，请确认 Ollama 服务已启动并已拉取 bge-m3 embedding 模型。"
        )
    else:
        reply = "这个问题需要进一步核对资料。请补充服务器、角色 ID 和具体问题描述。"
    return {
        **state,
        "reply": reply,
        "sources": [],
        "handoff": False,
    }


def generate_knowledge_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在整理知识库结果")
    results = state.get("knowledge_results", [])
    if not results:
        return generate_no_knowledge_reply(state)

    result = results[0]
    return {
        **state,
        "reply": result.content,
        "sources": [
            ChatSource(
                title=result.title,
                source_type="knowledge_base",
                reference=result.reference,
            )
        ],
        "handoff": False,
    }


def route_final_reply(state: CustomerServiceState) -> Literal["llm", "final"]:
    if state.get("use_llm_final_reply") and state.get("llm_client") is not None:
        return "llm"
    return "final"


async def generate_llm_final_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在生成回复")
    llm_client = state.get("llm_client")
    if llm_client is None:
        return state

    started_at = _trace_started(state, TraceEventType.LLM_STARTED, "final_reply")
    try:
        messages = _apply_context_budget_to_messages(
            state,
            final_reply_messages(state),
            protected_text=state.get("normalized_message", state.get("message", "")),
            priority_markers=["工具结果：", "结构化工具数据：", "来源："],
        )
        response = await llm_client.generate_reply(messages)
    except PromptNotFoundError:
        _trace_finished(
            state,
            TraceEventType.LLM_FINISHED,
            "final_reply",
            started_at,
            error="PromptNotFoundError",
        )
        logger.exception("Prompt loading failed during LLM final reply; session_id=%s", state.get("session_id"))
        raise
    except Exception as exc:
        _trace_finished(
            state,
            TraceEventType.LLM_FINISHED,
            "final_reply",
            started_at,
            error=type(exc).__name__,
        )
        logger.exception("LLM final reply failed; session_id=%s", state.get("session_id"))
        _emit_status(state, f"大模型生成失败，保留工具结果回复：{type(exc).__name__}")
        return state

    next_state = {**state, "reply": response.content}
    _trace_finished(next_state, TraceEventType.LLM_FINISHED, "final_reply", started_at)
    return next_state


def finalize_response(state: CustomerServiceState) -> CustomerServiceState:
    return state


def _emit_status(state: CustomerServiceState, message: str) -> None:
    status_queue = state.get("status_queue")
    if status_queue is not None:
        status_queue.put_nowait(message)


def _apply_context_budget_to_messages(
    state: CustomerServiceState,
    messages: list[dict[str, str]],
    *,
    protected_text: str,
    priority_markers: list[str] | None = None,
) -> list[dict[str, str]]:
    settings = get_settings()
    result = apply_context_budget(
        messages,
        ContextBudget(
            max_tokens=settings.llm_context_max_tokens,
            reserved_reply_tokens=settings.llm_context_reserved_reply_tokens,
        ),
        protected_text=protected_text,
        priority_markers=priority_markers,
    )
    state["context_budget_result"] = result
    return result.messages


def _trace_started(
    state: CustomerServiceState,
    event_type: TraceEventType,
    name: str,
    metadata: dict[str, object] | None = None,
) -> float | None:
    trace = state.get("agent_trace")
    if trace is None:
        return None
    return trace.record_started(event_type, name, metadata)


def _trace_finished(
    state: CustomerServiceState,
    event_type: TraceEventType,
    name: str,
    started_at: float | None,
    metadata: dict[str, object] | None = None,
    error: str = "",
) -> None:
    trace = state.get("agent_trace")
    if trace is None:
        return
    trace.record_finished(event_type, name, started_at, metadata, error=error)


def _record_conversation_exchange(session_id: str, user_message: str, assistant_reply: str) -> None:
    memory = get_conversation_memory()
    memory.append_message(session_id, "user", user_message)
    memory.append_message(session_id, "assistant", assistant_reply)


def _write_chat_checkpoint_safely(
    settings: Settings,
    *,
    session_id: str,
    player_id: str | None,
    message: str,
    response: ChatResponse,
    final_state: CustomerServiceState | None = None,
) -> None:
    try:
        write_chat_checkpoint_event(
            settings,
            session_id=session_id,
            player_id=player_id,
            message=message,
            response=response,
            final_state=final_state,
        )
    except Exception:
        logger.exception("Agent checkpoint writing failed; session_id=%s", session_id)


@lru_cache
def _compiled_graph():
    return build_customer_service_graph()


def _tool_execution_context(state: CustomerServiceState) -> ToolExecutionContext:
    return ToolExecutionContext(
        settings=get_settings(),
        message=state.get("normalized_message", state.get("message", "")),
        emit_status=lambda text: _emit_status(state, text),
        player_data_tools_factory=build_player_data_tools,
        map_agent_runner=run_map_agent,
        enforce_dependencies=False,
        agent_trace=state.get("agent_trace"),
    )


def _wrap_request_llm_client(
    client: LLMClientProtocol | None,
    *,
    session_id: str,
    provider_name: str,
    agent_trace: AgentTrace,
    usage_summary: LLMUsageSummary,
    settings: Settings,
) -> LLMClientProtocol | None:
    if client is None:
        return None
    input_price, output_price = _llm_token_prices_for_provider(settings, provider_name)
    return wrap_llm_client(
        client,
        LLMCallContext(
            provider=provider_name,
            model=_llm_context_model(client),
            session_id=session_id,
            agent_trace=agent_trace,
            usage_summary=usage_summary,
            input_token_price_per_1k=input_price,
            output_token_price_per_1k=output_price,
        ),
    )


def _llm_context_provider(settings, *, requested_provider: str | None, injected: bool) -> str:
    if injected:
        return "injected"
    return (
        requested_provider
        or settings.llm_default_provider
        or settings.llm_provider
        or "unknown"
    ).strip().lower()


def _llm_context_model(client: LLMClientProtocol) -> str:
    model = getattr(client, "model", "")
    if isinstance(model, str) and model:
        return model
    wrapped = getattr(client, "wrapped", None)
    if wrapped is not None:
        nested_model = getattr(wrapped, "model", "")
        if isinstance(nested_model, str) and nested_model:
            return nested_model
    return ""


def _llm_token_prices_for_provider(settings: Settings, provider_name: str) -> tuple[float, float]:
    normalized_provider = provider_name.strip().lower()
    if normalized_provider == "qwen":
        return (
            settings.qwen_input_token_price_per_1k,
            settings.qwen_output_token_price_per_1k,
        )
    if normalized_provider == "deepseek":
        return (
            settings.deepseek_input_token_price_per_1k,
            settings.deepseek_output_token_price_per_1k,
        )
    return 0.0, 0.0


def _player_profile_decision_for_tool_call(state: CustomerServiceState) -> AgentDecision:
    decision = state.get("llm_decision")
    if decision is not None and decision.action == AgentAction.MYSQL_PLAYER_PROFILE:
        player_id = _player_id_for_tool_call(state)
        if player_id and not (decision.arguments or {}).get("player_id"):
            return AgentDecision(
                action=decision.action,
                reason=decision.reason,
                arguments={**(decision.arguments or {}), "player_id": player_id},
                final_task=decision.final_task,
                direct_reply=decision.direct_reply,
            )
        return decision

    player_id = _player_id_for_tool_call(state)
    return AgentDecision(
        action=AgentAction.MYSQL_PLAYER_PROFILE,
        reason=decision.reason if decision is not None else "查询玩家资料",
        arguments={"player_id": player_id} if player_id else {},
        final_task=decision.final_task if decision is not None else "",
    )


def _players_list_decision_for_tool_call(state: CustomerServiceState) -> AgentDecision:
    decision = state.get("llm_decision")
    if decision is not None and decision.action == AgentAction.MYSQL_PLAYERS_LIST:
        return decision

    return AgentDecision(
        action=AgentAction.MYSQL_PLAYERS_LIST,
        reason="查询玩家列表",
        arguments={"limit": _players_limit_for_tool_call(state)},
        final_task=decision.final_task if decision is not None else "",
    )


def _player_data_result_from_execution(execution: ToolExecutionResult) -> PlayerDataResult:
    if isinstance(execution.output, PlayerDataResult):
        return execution.output

    if (
        execution.status == ToolExecutionStatus.INVALID_ARGUMENTS
        and execution.action == AgentAction.MYSQL_PLAYER_PROFILE
    ):
        return PlayerDataResult(
            status=PlayerDataStatus.NOT_FOUND,
            summary="请先提供玩家 ID，才能查询玩家资料。",
        )

    if execution.status == ToolExecutionStatus.MISSING_DEPENDENCY:
        return PlayerDataResult(
            status=PlayerDataStatus.DISABLED,
            summary="玩家数据查询尚未启用，请配置 MySQL 后再查询。",
        )

    return PlayerDataResult(
        status=PlayerDataStatus.UNAVAILABLE,
        summary=execution.error or "玩家数据暂时无法查询，请稍后重试或转人工客服处理。",
    )


def _map_agent_result_from_execution(
    execution: ToolExecutionResult,
    decision: AgentDecision,
) -> MapAgentResult:
    if isinstance(execution.output, MapAgentResult):
        return execution.output

    if execution.status == ToolExecutionStatus.INVALID_ARGUMENTS:
        map_result = MapToolResult(
            status=MapToolStatus.INVALID_REQUEST,
            summary=execution.error or "地图查询参数不合法，请补充更准确的信息。",
        )
    elif execution.status == ToolExecutionStatus.MISSING_DEPENDENCY:
        map_result = MapToolResult(
            status=MapToolStatus.DISABLED,
            summary="地图查询尚未启用，请配置高德地图 MCP 后再查询。",
        )
    else:
        map_result = MapToolResult(
            status=MapToolStatus.UNAVAILABLE,
            summary=execution.error or "地图查询暂时不可用，请稍后再试或转人工客服。",
        )
    return MapAgentResult(decision=decision, map_result=map_result)


def _player_id_for_tool_call(state: CustomerServiceState) -> str | None:
    decision = state.get("llm_decision")
    if decision is not None and decision.arguments:
        raw_player_id = decision.arguments.get("player_id")
        if isinstance(raw_player_id, str) and raw_player_id.strip():
            return raw_player_id.strip()

    return state.get("player_id")


def _presentation_plan_for_state(state: CustomerServiceState) -> PresentationPlan:
    decision = state.get("map_decision") or state.get("llm_decision")
    return build_presentation_plan(
        state.get("normalized_message", state.get("message", "")),
        conversation_history=state.get("conversation_history", []),
        decision_arguments=decision.arguments if decision is not None else None,
    )


def _combined_player_map_summary(state: CustomerServiceState, map_summary: str) -> str:
    player_data_result = state.get("player_data_result")
    if player_data_result is None:
        return map_summary
    return f"{player_data_result.summary}\n{map_summary}"


def _players_limit_for_tool_call(state: CustomerServiceState) -> int:
    decision = state.get("llm_decision")
    if decision is None or not decision.arguments:
        return 100

    raw_limit = decision.arguments.get("limit")
    if isinstance(raw_limit, int):
        return raw_limit
    if isinstance(raw_limit, str) and raw_limit.isdigit():
        return int(raw_limit)
    return 100


def _map_decision_for_tool_call(state: CustomerServiceState) -> AgentDecision:
    decision = state.get("map_decision") or state.get("llm_decision")
    if decision is not None and is_map_decision(decision):
        return decision
    return AgentDecision(
        action=AgentAction.AMAP_PLACE_SEARCH,
        reason="本地规则：地图类问题默认使用地点搜索",
        arguments={"keywords": state.get("normalized_message", state.get("message", ""))},
    )


def _current_plan_decision(state: CustomerServiceState) -> AgentDecision | None:
    plan = state.get("agent_plan")
    if plan is None:
        return None
    index = state.get("plan_step_index", 0)
    if index < 0 or index >= len(plan.steps):
        return None
    return plan.steps[index].to_decision(fallback_final_task=plan.final_task)


def _complete_current_plan_step(state: CustomerServiceState) -> CustomerServiceState:
    plan = state.get("agent_plan")
    if plan is None:
        return state

    index = state.get("plan_step_index", 0)
    if index < 0 or index >= len(plan.steps):
        return state

    step = plan.steps[index]
    completed_steps = [
        *state.get("completed_plan_steps", []),
        {
            "index": index,
            "action": str(step.action),
            "reason": step.reason,
            "status": "completed",
        },
    ]
    return {
        **state,
        "plan_step_index": index + 1,
        "completed_plan_steps": completed_steps,
    }
