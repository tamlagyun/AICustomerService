from pathlib import Path

from app.knowledge_base import KnowledgeBaseSearch
from app.rag.chroma_store import KnowledgeVectorIndexer
from app.rag.local_vector import LocalVectorKnowledgeIndex, VectorChunk


class FakeEmbeddingProvider:
    model = "bge-m3"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        return [[float(index + 1), 0.5] for index, _ in enumerate(texts)]


class FakeChromaStore:
    collection_name = "customer_service_knowledge"

    def __init__(self) -> None:
        self.rebuilt_chunks = []
        self.rebuilt_embeddings: list[list[float]] = []
        self.rebuilt_embedding_model = ""
        self.rebuilt_index_metadata = {}

    def rebuild(self, *, chunks, embeddings, embedding_model: str, index_metadata=None):
        self.rebuilt_chunks = chunks
        self.rebuilt_embeddings = embeddings
        self.rebuilt_embedding_model = embedding_model
        self.rebuilt_index_metadata = index_metadata or {}


def test_vector_search_finds_semantically_similar_recharge_question(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    vector_dir = tmp_path / "vectors"
    kb_dir.mkdir()
    (kb_dir / "recharge.md").write_text(
        "# 充值问题\n\n## 充值未到账怎么办\n\n请提供订单号、充值时间、服务器和角色 ID。",
        encoding="utf-8",
    )

    search = KnowledgeBaseSearch(kb_dir, retrieval_mode="vector", vector_store_dir=vector_dir)

    results = search.search("我充钱了但是没到账", limit=1)

    assert len(results) == 1
    assert results[0].title == "充值未到账怎么办"
    assert results[0].reference == "recharge.md#充值未到账怎么办"


def test_knowledge_vector_indexer_rebuilds_chroma_from_knowledge_chunks(
    tmp_path: Path,
) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "recharge.md").write_text(
        "# 充值问题\n\n## 充值未到账怎么办\n\n请提供订单号、充值时间、服务器和角色 ID。",
        encoding="utf-8",
    )
    embedding_provider = FakeEmbeddingProvider()
    chroma_store = FakeChromaStore()

    result = KnowledgeVectorIndexer(
        root_dir=kb_dir,
        embedding_provider=embedding_provider,
        store=chroma_store,
    ).rebuild()

    assert result.status == "rebuilt"
    assert result.chunk_count == 1
    assert result.collection_name == "customer_service_knowledge"
    assert result.embedding_model == "bge-m3"
    assert embedding_provider.inputs == [
        "充值未到账怎么办\n请提供订单号、充值时间、服务器和角色 ID。"
    ]
    assert chroma_store.rebuilt_embedding_model == "bge-m3"
    assert chroma_store.rebuilt_index_metadata["embedding_model"] == "bge-m3"
    assert chroma_store.rebuilt_index_metadata["file_count"] == 1
    assert chroma_store.rebuilt_embeddings == [[1.0, 0.5]]
    assert chroma_store.rebuilt_chunks[0].reference == "recharge.md#充值未到账怎么办"


def test_vector_index_refreshes_when_knowledge_file_changes(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    vector_dir = tmp_path / "vectors"
    kb_dir.mkdir()
    kb_file = kb_dir / "service.md"
    kb_file.write_text(
        "# 充值问题\n\n## 充值未到账怎么办\n\n请提供订单号、充值时间、服务器和角色 ID。",
        encoding="utf-8",
    )
    search = KnowledgeBaseSearch(kb_dir, retrieval_mode="vector", vector_store_dir=vector_dir)
    assert search.search("我充钱没到账", limit=1)[0].title == "充值未到账怎么办"

    kb_file.write_text(
        "# 账号问题\n\n## 账号被封禁怎么办\n\n封禁申诉需要转人工处理。",
        encoding="utf-8",
    )

    refreshed_results = search.search("账号封号怎么申诉", limit=1)

    assert refreshed_results[0].title == "账号被封禁怎么办"
    assert (vector_dir / "knowledge_base_vector_index.json").is_file()


def test_local_vector_index_orders_chunks_by_cosine_similarity(tmp_path: Path) -> None:
    index = LocalVectorKnowledgeIndex(root_dir=tmp_path, index_dir=tmp_path / "vectors")
    chunks = [
        VectorChunk(title="充值未到账怎么办", content="请提供订单号和充值时间。", reference="a.md#充值"),
        VectorChunk(title="账号被封禁怎么办", content="封禁申诉需要转人工。", reference="b.md#封禁"),
    ]

    results = index.search("我充钱了但是没到账", chunks, limit=2)

    assert [result.chunk.title for result in results] == [
        "充值未到账怎么办",
        "账号被封禁怎么办",
    ]
    assert results[0].score > results[1].score
