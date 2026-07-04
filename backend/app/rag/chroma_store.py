from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.rag.local_vector import VectorChunk

logger = logging.getLogger(__name__)


class EmbeddingProviderError(RuntimeError):
    pass


class ChromaUnavailableError(RuntimeError):
    pass


class ChromaIndexNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class ChromaKnowledgeHit:
    chunk: VectorChunk
    score: float


@dataclass(frozen=True)
class KnowledgeVectorRebuildResult:
    status: str
    chunk_count: int
    collection_name: str
    embedding_model: str
    message: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class OllamaEmbeddingProvider:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise EmbeddingProviderError(f"Ollama embedding request failed: {exc}") from exc

        embeddings = payload.get("embeddings")
        if embeddings is None and "embedding" in payload:
            embeddings = [payload["embedding"]]
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingProviderError("Ollama embedding response shape is invalid")

        parsed: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list):
                raise EmbeddingProviderError("Ollama embedding item is invalid")
            parsed.append([float(value) for value in embedding])
        return parsed


class ChromaKnowledgeStore:
    def __init__(
        self,
        *,
        persist_dir: str | Path,
        collection_name: str,
        min_score: float,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.min_score = min_score

    def rebuild(
        self,
        *,
        chunks: list[VectorChunk],
        embeddings: list[list[float]],
        embedding_model: str,
        index_metadata: dict[str, str | int] | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        client = self._client()
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._delete_collection_if_exists(client)
            collection = client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata=index_metadata or {"embedding_model": embedding_model},
            )
            if not chunks:
                return

            collection.add(
                ids=[_chunk_id(chunk, index) for index, chunk in enumerate(chunks)],
                documents=[_document_text(chunk) for chunk in chunks],
                metadatas=[
                    {
                        "title": chunk.title,
                        "reference": chunk.reference,
                    }
                    for chunk in chunks
                ],
                embeddings=embeddings,
            )
        finally:
            _close_chroma_client(client)

    def search(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[ChromaKnowledgeHit]:
        client = self._client()
        try:
            collection = self._existing_collection(client)
            if collection.count() == 0:
                raise ChromaIndexNotReady("向量知识库尚未建立")

            payload = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            documents = _first_result_list(payload.get("documents"))
            metadatas = _first_result_list(payload.get("metadatas"))
            distances = _first_result_list(payload.get("distances"))

            hits: list[ChromaKnowledgeHit] = []
            for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
                if not isinstance(metadata, dict):
                    continue
                score = _score_from_distance(distance)
                if score < self.min_score:
                    continue
                hits.append(
                    ChromaKnowledgeHit(
                        chunk=VectorChunk(
                            title=str(metadata.get("title", "")),
                            content=str(document),
                            reference=str(metadata.get("reference", "")),
                        ),
                        score=score,
                    )
                )
            return hits
        finally:
            _close_chroma_client(client)

    def _existing_collection(self, client):
        try:
            return client.get_collection(name=self.collection_name, embedding_function=None)
        except TypeError:
            try:
                return client.get_collection(name=self.collection_name)
            except Exception as exc:
                raise ChromaIndexNotReady("向量知识库尚未建立") from exc
        except Exception as exc:
            raise ChromaIndexNotReady("向量知识库尚未建立") from exc

    def existing_collection(self):
        client = self._client()
        try:
            return self._existing_collection(client)
        except Exception:
            _close_chroma_client(client)
            raise

    def _delete_collection_if_exists(self, client) -> None:
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            return

    def _client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise ChromaUnavailableError("Chroma 未安装，请先安装后端依赖。") from exc
        return chromadb.PersistentClient(path=str(self.persist_dir))


class KnowledgeVectorIndexer:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        embedding_provider: OllamaEmbeddingProvider,
        store: ChromaKnowledgeStore,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.embedding_provider = embedding_provider
        self.store = store

    def rebuild(self) -> KnowledgeVectorRebuildResult:
        chunks = self._load_chunks()
        vector_chunks = [_to_vector_chunk(chunk) for chunk in chunks]
        embeddings = self.embedding_provider.embed_texts([_document_text(chunk) for chunk in vector_chunks])
        from app.rag.vector_health import build_vector_index_metadata

        index_metadata = build_vector_index_metadata(
            knowledge_base_dir=self.root_dir,
            embedding_model=self.embedding_provider.model,
            collection_name=self.store.collection_name,
        )
        self.store.rebuild(
            chunks=vector_chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_provider.model,
            index_metadata=index_metadata.to_chroma_metadata(),
        )
        return KnowledgeVectorRebuildResult(
            status="rebuilt",
            chunk_count=len(vector_chunks),
            collection_name=self.store.collection_name,
            embedding_model=self.embedding_provider.model,
            message=f"已重建 {len(vector_chunks)} 个知识片段。",
        )

    def _load_chunks(self):
        from app.knowledge_base import KnowledgeBaseSearch

        return KnowledgeBaseSearch(self.root_dir, retrieval_mode="keyword").load_chunks()


def rebuild_knowledge_vector_index(settings: Settings | None = None) -> dict[str, str | int]:
    selected_settings = settings or get_settings()
    provider = OllamaEmbeddingProvider(
        base_url=selected_settings.ollama_base_url,
        model=selected_settings.ollama_embedding_model,
        timeout_seconds=selected_settings.ollama_embedding_timeout_seconds,
    )
    store = ChromaKnowledgeStore(
        persist_dir=selected_settings.chroma_persist_dir,
        collection_name=selected_settings.chroma_collection_name,
        min_score=selected_settings.chroma_min_score,
    )
    try:
        return KnowledgeVectorIndexer(
            root_dir=selected_settings.knowledge_base_dir,
            embedding_provider=provider,
            store=store,
        ).rebuild().to_dict()
    except Exception as exc:
        logger.exception("Knowledge vector index rebuild failed")
        return KnowledgeVectorRebuildResult(
            status="failed",
            chunk_count=0,
            collection_name=selected_settings.chroma_collection_name,
            embedding_model=selected_settings.ollama_embedding_model,
            message=f"知识库向量库重建失败：{type(exc).__name__}: {exc}",
        ).to_dict()


def _to_vector_chunk(chunk) -> VectorChunk:
    return VectorChunk(title=chunk.title, content=chunk.content, reference=chunk.reference)


def _document_text(chunk: VectorChunk) -> str:
    return f"{chunk.title}\n{chunk.content}".strip()


def _chunk_id(chunk: VectorChunk, index: int) -> str:
    digest = hashlib.sha1(f"{index}:{chunk.reference}".encode("utf-8")).hexdigest()
    return digest


def _first_result_list(value: Any) -> list:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else []


def _score_from_distance(distance: object) -> float:
    try:
        numeric_distance = max(float(distance), 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + numeric_distance)


def _close_chroma_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
    clear_system_cache = getattr(client, "clear_system_cache", None)
    if callable(clear_system_cache):
        clear_system_cache()
