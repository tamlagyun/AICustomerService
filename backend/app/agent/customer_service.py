import asyncio
from functools import lru_cache
import json
from typing import Any, AsyncIterator, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.decision import AgentAction, AgentDecision
from app.avatar_generation import AvatarGenerationResult, build_avatar_generator
from app.config import get_settings
from app.conversation_memory import ConversationMessage, get_conversation_memory
from app.knowledge_base import KnowledgeBaseSearch, KnowledgeChunk
from app.llm import LLMClientProtocol, build_llm_client
from app.map_tools import MapToolResult, build_map_tools
from app.player_data import PlayerDataResult, PlayerDataStatus, build_player_data_tools
from app.safety import SafetyAction, SafetyDecision, analyze_safety, redact_sensitive_text
from app.schemas import ChatImage, ChatResponse, ChatSource, ChatTable
from app.table_adapter import tables_for_map_result, tables_for_player_data_result

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
    question_type: QuestionType
    safety_decision: SafetyDecision
    llm_client: LLMClientProtocol | None
    llm_decision: AgentDecision
    use_llm_final_reply: bool
    knowledge_results: list[KnowledgeChunk]
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


def build_customer_service_graph():
    workflow = StateGraph(CustomerServiceState)
    workflow.add_node("analyze_safety", analyze_safety_node)
    workflow.add_node("decide_action_with_llm", decide_action_with_llm)
    workflow.add_node("classify_question", classify_question)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("retrieve_player_data", retrieve_player_data)
    workflow.add_node("retrieve_players_list", retrieve_players_list)
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
            "allow": "decide_action_with_llm",
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
            "player_data": "generate_player_data_reply",
        },
    )
    workflow.add_edge("retrieve_players_list", "generate_player_data_reply")
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
    llm_client: LLMClientProtocol | None = None,
    status_queue: asyncio.Queue[str] | None = None,
) -> ChatResponse:
    memory = get_conversation_memory()
    capability_reply = _system_capability_reply(message)
    if capability_reply is not None:
        response = ChatResponse(reply=capability_reply)
        _record_conversation_exchange(session_id, message, response.reply)
        return response

    final_state = await _compiled_graph().ainvoke(
        {
            "session_id": session_id,
            "player_id": player_id,
            "message": message,
            "conversation_history": memory.get_recent_messages(session_id),
            "llm_client": llm_client if llm_client is not None else build_llm_client(model_provider),
            "status_queue": status_queue,
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
    llm_client: LLMClientProtocol | None = None,
) -> AsyncIterator[dict[str, Any]]:
    yield {"event": "status", "data": {"message": "正在分析问题"}}

    base_llm_client = llm_client if llm_client is not None else build_llm_client(model_provider)
    token_queue: asyncio.Queue[str] = asyncio.Queue()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    streaming_llm_client = (
        _StreamingLLMClient(base_llm_client, token_queue) if base_llm_client is not None else None
    )

    task = asyncio.create_task(
        run_customer_service_agent(
            session_id=session_id,
            player_id=player_id,
            message=message,
            model_provider=model_provider,
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
    normalized_message = state["message"].strip()
    return {
        **state,
        "normalized_message": normalized_message,
        "safety_decision": analyze_safety(normalized_message),
    }


def route_safety(state: CustomerServiceState) -> Literal["allow", "handoff", "refuse"]:
    action = state["safety_decision"].action
    if action == SafetyAction.REFUSE:
        return "refuse"
    if action == SafetyAction.HANDOFF:
        return "handoff"
    return "allow"


async def decide_action_with_llm(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在判断是否需要调用工具")
    llm_client = state.get("llm_client")
    if llm_client is None:
        _emit_status(state, "正在使用本地规则决策")
        return state

    try:
        _emit_status(state, "正在请求大模型决策")
        decision = await llm_client.decide_action(
            _decision_messages(
                state["normalized_message"],
                state.get("conversation_history", []),
            )
        )
    except Exception:
        return state

    return {**state, "llm_decision": decision}


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
    if decision.action == AgentAction.KNOWLEDGE_BASE:
        return "knowledge"
    if decision.action == AgentAction.MYSQL_PLAYER_PROFILE:
        return "player_data"
    if decision.action == AgentAction.MYSQL_PLAYERS_LIST:
        return "players_list"
    if decision.action == AgentAction.AVATAR_GENERATE:
        return "avatar"
    if decision.action in {
        AgentAction.AMAP_PLACE_SEARCH,
        AgentAction.AMAP_GEO,
        AgentAction.AMAP_ROUTE,
        AgentAction.AMAP_NAVIGATION,
        AgentAction.AMAP_WEATHER,
    }:
        return "map"
    return "fallback"


def classify_question(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在使用本地规则决策")
    normalized_message = state["normalized_message"]
    question_type: QuestionType = "general"

    if _looks_like_players_list_query(normalized_message):
        question_type = "players_list"
    elif any(keyword in normalized_message for keyword in ["玩家资料", "玩家信息", "角色资料", "角色信息"]):
        question_type = "player_data"
    elif state.get("player_id") and any(keyword in normalized_message for keyword in ["资料", "信息"]):
        question_type = "player_data"
    elif any(keyword in normalized_message for keyword in ["充值", "不到账", "订单", "封禁"]):
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


def _looks_like_players_list_query(message: str) -> bool:
    normalized = message.lower()
    asks_many = any(keyword in message for keyword in ["所有", "全部", "列表", "全量"])
    asks_player_rows = any(keyword in message for keyword in ["玩家", "资料", "数据", "信息"])

    if "players" in normalized and asks_many:
        return True
    if "数据库" in message and asks_many and asks_player_rows:
        return True
    return any(keyword in message for keyword in ["所有玩家", "全部玩家", "玩家列表"])


def route_after_knowledge(state: CustomerServiceState) -> Literal["knowledge", "general", "fallback"]:
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
    results = KnowledgeBaseSearch(get_settings().knowledge_base_dir).search(
        state["normalized_message"],
        limit=1,
    )
    return {
        **state,
        "knowledge_results": results,
        "use_llm_final_reply": _should_use_llm_final_reply(state),
    }


def retrieve_player_data(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在查询 MySQL 玩家数据")
    result = build_player_data_tools().get_player_profile(_player_id_for_tool_call(state))
    return {
        **state,
        "player_data_result": result,
        "use_llm_final_reply": _should_use_llm_final_reply(state),
    }


def retrieve_players_list(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在查询 MySQL 玩家列表")
    result = build_player_data_tools().get_players(_players_limit_for_tool_call(state))
    return {
        **state,
        "player_data_result": result,
        "use_llm_final_reply": _should_use_llm_final_reply(state),
    }


async def retrieve_map_data(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在调用高德地图 MCP")
    decision = state.get("llm_decision")
    map_tools = build_map_tools()

    if decision is not None and decision.action == AgentAction.AMAP_GEO:
        result = await map_tools.geocode(
            _string_argument(state, "address"),
            city=_string_argument(state, "city"),
        )
    elif decision is not None and decision.action == AgentAction.AMAP_ROUTE:
        result = await map_tools.route(
            origin=_string_argument(state, "origin"),
            destination=_string_argument(state, "destination"),
            mode=_string_argument(state, "mode"),
            city=_string_argument(state, "city"),
            cityd=_string_argument(state, "cityd"),
        )
    elif decision is not None and decision.action == AgentAction.AMAP_NAVIGATION:
        result = await map_tools.navigation(
            destination=_string_argument(state, "destination"),
            destination_name=_string_argument(state, "destination_name")
            or _string_argument(state, "destination"),
            origin=_string_argument(state, "origin"),
            origin_name=_string_argument(state, "origin_name"),
            mode=_string_argument(state, "mode"),
            city=_string_argument(state, "city"),
        )
    elif decision is not None and decision.action == AgentAction.AMAP_WEATHER:
        result = await map_tools.weather(_string_argument(state, "city"))
    else:
        result = await map_tools.search_place(
            _map_keywords_for_tool_call(state),
            city=_string_argument(state, "city"),
            types=_string_argument(state, "types"),
        )

    return {
        **state,
        "map_result": result,
        "use_llm_final_reply": _should_use_llm_final_reply(state),
    }


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
    return {
        **state,
        "reply": player_data_result.summary,
        "sources": [],
        "tables": tables_for_player_data_result(
            player_data_result,
            tool_name=_tool_name_for_prompt(state),
        ),
        "handoff": False,
    }


def generate_map_reply(state: CustomerServiceState) -> CustomerServiceState:
    _emit_status(state, "正在整理地图结果")
    map_result = state["map_result"]
    sources = []
    tool_name = _map_tool_name(map_result)
    if tool_name:
        sources.append(
            ChatSource(
                title="高德地图 MCP",
                source_type="amap_mcp",
                reference=tool_name,
            )
        )

    return {
        **state,
        "reply": map_result.summary,
        "sources": sources,
        "tables": tables_for_map_result(map_result),
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
        return {
            **state,
            "reply": reply,
            "sources": [],
            "images": [],
            "handoff": False,
        }

    avatar_result = build_avatar_generator().generate_player_avatar(
        player_data_result.data,
        session_id=state["session_id"],
    )
    images = (
        [ChatImage(url=avatar_result.url, alt=avatar_result.alt)]
        if avatar_result.url and avatar_result.alt
        else []
    )
    return {
        **state,
        "avatar_result": avatar_result,
        "reply": avatar_result.summary,
        "sources": [],
        "images": images,
        "handoff": False,
    }


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
    return {
        **state,
        "reply": "这个问题需要进一步核对资料。请补充服务器、角色 ID 和具体问题描述。",
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

    try:
        response = await llm_client.generate_reply(_final_reply_messages(state))
    except Exception:
        return state

    return {**state, "reply": response.content}


def finalize_response(state: CustomerServiceState) -> CustomerServiceState:
    return state


def _emit_status(state: CustomerServiceState, message: str) -> None:
    status_queue = state.get("status_queue")
    if status_queue is not None:
        status_queue.put_nowait(message)


def _record_conversation_exchange(session_id: str, user_message: str, assistant_reply: str) -> None:
    memory = get_conversation_memory()
    memory.append_message(session_id, "user", user_message)
    memory.append_message(session_id, "assistant", assistant_reply)


def _format_conversation_history(messages: list[ConversationMessage]) -> str:
    if not messages:
        return "无"

    lines = []
    for message in messages:
        role_label = "玩家" if message.role == "user" else "客服"
        lines.append(f"{role_label}：{message.content}")
    return "\n".join(lines)


@lru_cache
def _compiled_graph():
    return build_customer_service_graph()


def _decision_messages(
    message: str,
    conversation_history: list[ConversationMessage],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是游戏客服 Agent 的决策器。只允许输出 JSON，不要输出额外文字。"
                "可选 action: knowledge_base, mysql_player_profile, mysql_players_list, "
                "avatar_generate, amap_place_search, amap_geo, amap_route, amap_navigation, amap_weather, "
                "ask_clarification, handoff, direct_answer。"
                "不能生成 SQL。数据库查询只能选择 mysql_player_profile 或 mysql_players_list。"
                "如果需要玩家资料，选择 mysql_player_profile，并在 arguments.player_id 中填写玩家 ID。"
                "如果用户要求查询 players 表所有数据、数据库中所有资料、数据库全部数据、全部玩家、玩家列表，选择 mysql_players_list。"
                "mysql_players_list 可选 arguments.limit，默认 100，最大 1000。"
                "如果用户要求生成头像、形象图、个性头像，选择 avatar_generate；该动作会先通过后端工具查询玩家资料，再生成本地 PNG 头像。"
                "avatar_generate 需要 arguments.player_id；如果历史对话或当前问题没有明确玩家 ID，应选择 ask_clarification。"
                "如果玩家询问地点、地址、在哪里、附近有什么地点，选择 amap_place_search，"
                "arguments.keywords 填地点关键词，arguments.city 可填城市，arguments.types 可填 POI 类型。"
                "如果玩家要求把地址或地名解析为经纬度，选择 amap_geo，arguments.address 必填，arguments.city 可选。"
                "如果玩家要求路线、距离或怎么去，选择 amap_route，"
                "arguments.origin 和 arguments.destination 填起终点地址或高德经纬度，arguments.mode 可填 driving、walking、bicycling 或 transit。"
                "如果玩家要求打开导航、给导航链接、导航到目的地，选择 amap_navigation，"
                "arguments.destination 填目的地地址或高德经纬度；如果有起点则填 arguments.origin；mode 可填 driving、walking、bicycling 或 transit。"
                "如果玩家询问天气，选择 amap_weather，arguments.city 填城市名称或 adcode。"
                "如果路线或导航缺少起点但可使用当前位置作为起点，可以不填 origin；如果目的地缺失，选择 ask_clarification。"
                "玩家可能会用“玩家ID”“角色ID”“我的ID”“唯一Key”“ID”等说法表达玩家 ID。"
                "如果玩家要求查询后继续分析、总结或建议，把最终任务写入 final_task。"
                "如果玩家 ID 缺失或有多个候选导致歧义，选择 ask_clarification 并在 direct_reply 中反问。"
                "如果需要查询客服知识库，选择 knowledge_base。"
                "如果需要人工处理，选择 handoff。"
                "当前系统能力：前端聊天默认调用 /api/chat/stream，后端通过 SSE 流式输出 token；"
                "旧的 /api/chat REST 一次性返回接口仍保留用于兼容。"
                "如果玩家询问是否采用流式输出，应回答已经采用 SSE 流式输出。"
                "如果可以直接回复，选择 direct_answer 并填写 direct_reply。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"历史对话（最近 10 条）：\n{_format_conversation_history(conversation_history)}\n"
                f"当前玩家问题：{message}\n"
                "请结合历史对话和当前问题输出 JSON，例如："
                '{"action":"mysql_player_profile","arguments":{"player_id":"1"},'
                '"final_task":"根据玩家资料和 desc 字段分析总结玩家个性",'
                '"reason":"需要查询玩家资料后继续分析","direct_reply":""}'
            ),
        },
    ]


def _final_reply_messages(state: CustomerServiceState) -> list[dict[str, str]]:
    sources = state.get("sources", [])
    source_text = "\n".join(
        f"- {source.title}: {source.reference}" for source in sources
    ) or "无"
    decision = state.get("llm_decision")
    final_task = decision.final_task if decision is not None else ""
    tool_data = _tool_data_for_prompt(state)
    conversation_history = _format_conversation_history(state.get("conversation_history", []))
    return [
        {
            "role": "system",
            "content": (
                "你是游戏客服 AI。根据玩家问题和后端工具结果生成简洁、准确的中文回复。"
                "不要编造工具结果中没有的信息。不要承诺未验证的补偿、退款或封禁解除。"
                "如果玩家要求分析个性，只能基于工具结果中的 desc 字段和基础资料分析。"
                "如果工具结果是本地 PNG 头像，只能说明已生成可预览的本地头像，不要声称已调用真实生图模型。"
                "如果工具结果来自高德地图 MCP 或高德 URI API，只能基于地图工具结果回答地点、地址、路线、导航或天气，不要编造未返回的门店、电话、营业时间或天气。"
                "如果 desc 或相关字段不足以支持结论，必须明确说明信息不足。"
                "如果工具结果显示无法查询或未启用，如实说明并引导转人工或补充信息。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"历史对话（最近 10 条）：\n{conversation_history}\n"
                f"玩家问题：{state.get('normalized_message', state.get('message', ''))}\n"
                f"最终任务：{final_task or '按玩家问题回复'}\n"
                f"工具结果：{state.get('reply', '')}\n"
                f"结构化工具数据：{tool_data}\n"
                f"来源：\n{source_text}\n"
                "请生成最终回复。"
            ),
        },
    ]


def _should_use_llm_final_reply(state: CustomerServiceState) -> bool:
    decision = state.get("llm_decision")
    return (
        state.get("llm_client") is not None
        and decision is not None
        and decision.action != AgentAction.FALLBACK
    )


def _player_id_for_tool_call(state: CustomerServiceState) -> str | None:
    decision = state.get("llm_decision")
    if decision is not None and decision.arguments:
        raw_player_id = decision.arguments.get("player_id")
        if isinstance(raw_player_id, str) and raw_player_id.strip():
            return raw_player_id.strip()

    return state.get("player_id")


def _tool_data_for_prompt(state: CustomerServiceState) -> str:
    tools: list[dict[str, object]] = []
    player_data_result = state.get("player_data_result")
    if player_data_result is not None:
        tools.append(
            {
                "tool": _tool_name_for_prompt(state),
                "status": player_data_result.status,
                "data": player_data_result.data,
            }
        )

    avatar_result = state.get("avatar_result")
    if avatar_result is not None:
        tools.append(
            {
                "tool": "avatar_generate",
                "status": avatar_result.status,
                "url": avatar_result.url,
                "alt": avatar_result.alt,
                "data": avatar_result.data,
            }
        )

    map_result = state.get("map_result")
    if map_result is not None:
        tools.append(
            {
                "tool": _map_tool_name(map_result) or "amap_mcp",
                "status": map_result.status,
                "data": map_result.data,
            }
        )

    if not tools:
        return "{}"

    return json.dumps({"tools": tools}, ensure_ascii=False)


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


def _tool_name_for_prompt(state: CustomerServiceState) -> str:
    decision = state.get("llm_decision")
    if decision is not None and decision.action == AgentAction.MYSQL_PLAYERS_LIST:
        return "mysql_players_list"
    if state.get("question_type") == "players_list":
        return "mysql_players_list"
    return "mysql_player_profile"


def _string_argument(state: CustomerServiceState, name: str) -> str | None:
    decision = state.get("llm_decision")
    if decision is None or not decision.arguments:
        return None

    value = decision.arguments.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _map_keywords_for_tool_call(state: CustomerServiceState) -> str:
    decision = state.get("llm_decision")
    if decision is not None and decision.arguments:
        keywords = decision.arguments.get("keywords")
        if isinstance(keywords, str) and keywords.strip():
            return keywords.strip()

    return state.get("normalized_message", state.get("message", ""))


def _map_tool_name(map_result: MapToolResult) -> str | None:
    if not map_result.data:
        return None
    tool_name = map_result.data.get("tool")
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    return None


class _StreamingLLMClient:
    def __init__(self, wrapped: LLMClientProtocol, token_queue: asyncio.Queue[str]) -> None:
        self.wrapped = wrapped
        self.token_queue = token_queue

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return await self.wrapped.decide_action(messages)

    async def generate_reply(self, messages: list[dict[str, str]]):
        content = ""
        stream_reply = getattr(self.wrapped, "stream_reply", None)
        if stream_reply is None:
            return await self.wrapped.generate_reply(messages)

        try:
            async for token in stream_reply(messages):
                content += token
                await self.token_queue.put(token)
        except Exception:
            return await self.wrapped.generate_reply(messages)

        from app.llm import LLMResponse

        return LLMResponse(content=content)

    async def stream_reply(self, messages: list[dict[str, str]]):
        async for token in self.wrapped.stream_reply(messages):
            yield token
