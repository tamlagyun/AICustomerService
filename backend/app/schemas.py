from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    player_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatSource(BaseModel):
    title: str
    source_type: str
    reference: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[ChatSource] = []
    handoff: bool = False
