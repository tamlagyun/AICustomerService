from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.rag.vector_health import (
    VectorHealthStatus,
    VectorIndexMetadata,
    compute_knowledge_base_fingerprint,
    evaluate_vector_health,
    get_vector_store_health,
)


class FakeCollection:
    def __init__(self, *, metadata: dict, count: int = 2) -> None:
        self.metadata = metadata
        self._count = count

    def count(self) -> int:
        return self._count


def test_compute_knowledge_base_fingerprint_tracks_md_and_html_files(tmp_path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("# A\n充值未到账", encoding="utf-8")
    (kb_dir / "b.html").write_text("<h1>B</h1>", encoding="utf-8")
    (kb_dir / "ignore.txt").write_text("ignore", encoding="utf-8")

    fingerprint = compute_knowledge_base_fingerprint(kb_dir)

    assert fingerprint.file_count == 2
    assert fingerprint.file_hash


def test_vector_health_ready_when_metadata_matches_current_files(tmp_path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("# A\n充值未到账", encoding="utf-8")
    fingerprint = compute_knowledge_base_fingerprint(kb_dir)
    metadata = VectorIndexMetadata.from_fingerprint(
        knowledge_base_dir=str(kb_dir),
        fingerprint=fingerprint,
        embedding_provider="ollama",
        embedding_model="bge-m3",
        collection_name="customer_service_knowledge",
    )

    health = evaluate_vector_health(
        collection=FakeCollection(metadata=metadata.to_chroma_metadata(), count=1),
        settings=Settings(
            knowledge_base_dir=str(kb_dir),
            chroma_collection_name="customer_service_knowledge",
            ollama_embedding_model="bge-m3",
        ),
    )

    assert health.status == VectorHealthStatus.READY
    assert health.document_count == 1
    assert health.metadata.file_hash == fingerprint.file_hash


def test_vector_health_stale_when_file_hash_changes(tmp_path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    kb_file = kb_dir / "a.md"
    kb_file.write_text("# A\n旧内容", encoding="utf-8")
    old_fingerprint = compute_knowledge_base_fingerprint(kb_dir)
    metadata = VectorIndexMetadata.from_fingerprint(
        knowledge_base_dir=str(kb_dir),
        fingerprint=old_fingerprint,
        embedding_provider="ollama",
        embedding_model="bge-m3",
        collection_name="customer_service_knowledge",
    )
    kb_file.write_text("# A\n新内容", encoding="utf-8")

    health = evaluate_vector_health(
        collection=FakeCollection(metadata=metadata.to_chroma_metadata(), count=1),
        settings=Settings(
            knowledge_base_dir=str(kb_dir),
            chroma_collection_name="customer_service_knowledge",
            ollama_embedding_model="bge-m3",
        ),
    )

    assert health.status == VectorHealthStatus.STALE
    assert "知识库文件已变更" in health.message


def test_get_vector_store_health_returns_not_ready_when_collection_missing(monkeypatch, tmp_path) -> None:
    class MissingCollectionStore:
        def existing_collection(self):
            raise __import__(
                "app.rag.chroma_store",
                fromlist=["ChromaIndexNotReady"],
            ).ChromaIndexNotReady("missing")

    health = get_vector_store_health(
        Settings(
            knowledge_base_dir=str(tmp_path),
            chroma_persist_dir=str(tmp_path / "chroma"),
            chroma_collection_name="customer_service_knowledge",
        ),
        store_factory=lambda settings: MissingCollectionStore(),
    )

    assert health.status == VectorHealthStatus.NOT_READY


def test_get_vector_store_health_returns_unavailable_when_chroma_unavailable(tmp_path) -> None:
    class UnavailableStore:
        def existing_collection(self):
            raise __import__(
                "app.rag.chroma_store",
                fromlist=["ChromaUnavailableError"],
            ).ChromaUnavailableError("missing package")

    health = get_vector_store_health(
        Settings(
            knowledge_base_dir=str(tmp_path),
            chroma_persist_dir=str(tmp_path / "chroma"),
            chroma_collection_name="customer_service_knowledge",
        ),
        store_factory=lambda settings: UnavailableStore(),
    )

    assert health.status == VectorHealthStatus.UNAVAILABLE


def test_vector_health_api_returns_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    def fake_health(settings):
        return {
            "status": "ready",
            "message": "向量库可用。",
            "collection_name": "customer_service_knowledge",
            "document_count": 2,
            "metadata": {
                "file_count": 1,
                "file_hash": "abc",
                "indexed_at": "2026-07-04T00:00:00+00:00",
                "embedding_provider": "ollama",
                "embedding_model": "bge-m3",
                "collection_name": "customer_service_knowledge",
            },
        }

    monkeypatch.setattr("app.main.get_vector_store_health_payload", fake_health)
    client = TestClient(app)

    response = client.get("/api/knowledge-base/vector-health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
