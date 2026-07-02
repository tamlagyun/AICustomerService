import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent import run_customer_service_agent, stream_customer_service_agent
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse

settings = get_settings()

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
            ):
                yield _format_sse(event["event"], event["data"])
        except Exception:
            yield _format_sse("error", {"message": "服务暂时不可用，请稍后重试。"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
