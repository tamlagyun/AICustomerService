from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    player_id: str | None = None
    model_provider: str | None = Field(default=None, max_length=32)
    message: str = Field(min_length=1, max_length=4000)


class ChatSource(BaseModel):
    title: str
    source_type: str
    reference: str


class ChatImage(BaseModel):
    url: str
    alt: str


class ChatTableColumn(BaseModel):
    key: str
    label: str


class ChatTable(BaseModel):
    title: str
    columns: list[ChatTableColumn]
    rows: list[dict[str, Any]]


class ChatResponse(BaseModel):
    reply: str
    sources: list[ChatSource] = Field(default_factory=list)
    handoff: bool = False
    images: list[ChatImage] = Field(default_factory=list)
    tables: list[ChatTable] = Field(default_factory=list)
