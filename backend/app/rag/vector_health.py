from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.rag.chroma_store import ChromaIndexNotReady, ChromaKnowledgeStore, ChromaUnavailableError


class VectorHealthStatus(StrEnum):
    READY = "ready"
    STALE = "stale"
    NOT_READY = "not_ready"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class KnowledgeBaseFingerprint:
    file_count: int
    file_hash: str


@dataclass(frozen=True)
class VectorIndexMetadata:
    knowledge_base_dir: str
    file_count: int
    file_hash: str
    indexed_at: str
    embedding_provider: str
    embedding_model: str
    collection_name: str

    @classmethod
    def from_fingerprint(
        cls,
        *,
        knowledge_base_dir: str,
        fingerprint: KnowledgeBaseFingerprint,
        embedding_provider: str,
        embedding_model: str,
        collection_name: str,
        indexed_at: str | None = None,
    ) -> VectorIndexMetadata:
        return cls(
            knowledge_base_dir=knowledge_base_dir,
            file_count=fingerprint.file_count,
            file_hash=fingerprint.file_hash,
            indexed_at=indexed_at or datetime.now(UTC).isoformat(),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )

    @classmethod
    def from_chroma_metadata(cls, metadata: dict[str, Any] | None) -> VectorIndexMetadata | None:
        if not isinstance(metadata, dict):
            return None
        try:
            return cls(
                knowledge_base_dir=str(metadata.get("knowledge_base_dir", "")),
                file_count=int(metadata.get("file_count", 0)),
                file_hash=str(metadata.get("file_hash", "")),
                indexed_at=str(metadata.get("indexed_at", "")),
                embedding_provider=str(metadata.get("embedding_provider", "")),
                embedding_model=str(metadata.get("embedding_model", "")),
                collection_name=str(metadata.get("collection_name", "")),
            )
        except (TypeError, ValueError):
            return None

    def to_chroma_metadata(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class VectorStoreHealth:
    status: VectorHealthStatus
    message: str
    collection_name: str
    document_count: int = 0
    metadata: VectorIndexMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "message": self.message,
            "collection_name": self.collection_name,
            "document_count": self.document_count,
            "metadata": self.metadata.to_chroma_metadata() if self.metadata else None,
        }


class VectorHealthStoreProtocol(Protocol):
    def existing_collection(self) -> Any:
        raise NotImplementedError


def compute_knowledge_base_fingerprint(root_dir: str | Path) -> KnowledgeBaseFingerprint:
    root = Path(root_dir)
    digest = hashlib.sha256()
    file_count = 0
    if not root.exists():
        return KnowledgeBaseFingerprint(file_count=0, file_hash="")

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".html", ".htm"}:
            continue
        file_count += 1
        relative_path = path.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return KnowledgeBaseFingerprint(
        file_count=file_count,
        file_hash=digest.hexdigest() if file_count else "",
    )


def build_vector_index_metadata(
    *,
    knowledge_base_dir: str | Path,
    embedding_model: str,
    collection_name: str,
) -> VectorIndexMetadata:
    return VectorIndexMetadata.from_fingerprint(
        knowledge_base_dir=str(Path(knowledge_base_dir)),
        fingerprint=compute_knowledge_base_fingerprint(knowledge_base_dir),
        embedding_provider="ollama",
        embedding_model=embedding_model,
        collection_name=collection_name,
    )


def evaluate_vector_health(collection: Any, settings: Settings) -> VectorStoreHealth:
    metadata = VectorIndexMetadata.from_chroma_metadata(getattr(collection, "metadata", None))
    document_count = _collection_count(collection)
    collection_name = settings.chroma_collection_name
    if metadata is None:
        return VectorStoreHealth(
            status=VectorHealthStatus.STALE,
            message="向量库缺少索引元数据，请重建。",
            collection_name=collection_name,
            document_count=document_count,
        )

    current_fingerprint = compute_knowledge_base_fingerprint(settings.knowledge_base_dir)
    if metadata.file_hash != current_fingerprint.file_hash or metadata.file_count != current_fingerprint.file_count:
        return VectorStoreHealth(
            status=VectorHealthStatus.STALE,
            message="知识库文件已变更，请重建向量库。",
            collection_name=collection_name,
            document_count=document_count,
            metadata=metadata,
        )

    if metadata.embedding_model != settings.ollama_embedding_model:
        return VectorStoreHealth(
            status=VectorHealthStatus.STALE,
            message="Embedding 模型配置已变化，请重建向量库。",
            collection_name=collection_name,
            document_count=document_count,
            metadata=metadata,
        )

    return VectorStoreHealth(
        status=VectorHealthStatus.READY,
        message="向量库可用。",
        collection_name=collection_name,
        document_count=document_count,
        metadata=metadata,
    )


def get_vector_store_health(
    settings: Settings | None = None,
    *,
    store_factory: Any | None = None,
) -> VectorStoreHealth:
    selected_settings = settings or get_settings()
    factory = store_factory or _default_store_factory
    store = factory(selected_settings)
    try:
        collection = store.existing_collection()
    except ChromaIndexNotReady:
        return VectorStoreHealth(
            status=VectorHealthStatus.NOT_READY,
            message="向量库尚未建立，请先重建。",
            collection_name=selected_settings.chroma_collection_name,
        )
    except ChromaUnavailableError as exc:
        return VectorStoreHealth(
            status=VectorHealthStatus.UNAVAILABLE,
            message=str(exc),
            collection_name=selected_settings.chroma_collection_name,
        )
    return evaluate_vector_health(collection, selected_settings)


def get_vector_store_health_payload(settings: Settings | None = None) -> dict[str, Any]:
    return get_vector_store_health(settings).to_dict()


def _default_store_factory(settings: Settings) -> ChromaKnowledgeStore:
    return ChromaKnowledgeStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        min_score=settings.chroma_min_score,
    )


def _collection_count(collection: Any) -> int:
    count = getattr(collection, "count", None)
    if not callable(count):
        return 0
    try:
        return int(count())
    except (TypeError, ValueError):
        return 0
