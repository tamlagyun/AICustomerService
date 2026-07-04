import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent import run_customer_service_agent, stream_customer_service_agent
from app.agent.checkpoint import CheckpointStore
from app.config import get_settings
from app.evaluations import EvaluationRunRequest, ensure_evaluation_enabled, list_evaluation_cases
from app.evaluations import run_evaluation_suite
from app.logging_config import configure_logging
from app.rag.chroma_store import rebuild_knowledge_vector_index
from app.rag.vector_health import get_vector_store_health_payload
from app.schemas import ChatRequest, ChatResponse

settings = get_settings()
configure_logging(settings)

app = FastAPI(title="Customer Service AI Agent API")
generated_dir = Path(__file__).resolve().parents[2] / "generated"
generated_dir.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/generated", StaticFiles(directory=generated_dir), name="generated")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await run_customer_service_agent(
        session_id=request.session_id,
        player_id=request.player_id,
        message=request.message,
        model_provider=request.model_provider,
        use_planner=request.use_planner,
        knowledge_source=request.knowledge_source,
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in stream_customer_service_agent(
                session_id=request.session_id,
                player_id=request.player_id,
                message=request.message,
                model_provider=request.model_provider,
                use_planner=request.use_planner,
                knowledge_source=request.knowledge_source,
            ):
                yield _format_sse(event["event"], event["data"])
        except Exception:
            yield _format_sse("error", {"message": "服务暂时不可用，请稍后重试。"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/evaluations/cases")
async def evaluation_cases() -> dict:
    ensure_evaluation_enabled()
    return list_evaluation_cases()


@app.post("/api/evaluations/run")
async def evaluation_run(request: EvaluationRunRequest) -> dict:
    ensure_evaluation_enabled()
    return await run_evaluation_suite(request)


@app.post("/api/knowledge-base/vector-index/rebuild")
async def knowledge_vector_index_rebuild() -> dict:
    return rebuild_knowledge_vector_index()


@app.get("/api/knowledge-base/vector-health")
async def knowledge_vector_health() -> dict:
    return get_vector_store_health_payload(get_settings())


@app.get("/api/checkpoints")
async def checkpoints(session_id: str | None = None, limit: int = 20) -> dict:
    current_settings = get_settings()
    if not current_settings.agent_checkpoint_enabled:
        raise HTTPException(status_code=403, detail="Agent checkpoint API is disabled")
    store = CheckpointStore(current_settings)
    return {
        "checkpoints": store.list_recent(session_id=session_id, limit=limit),
    }


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
