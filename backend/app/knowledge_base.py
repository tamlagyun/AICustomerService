from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re

from app.config import get_settings
from app.rag.chroma_store import ChromaKnowledgeStore, OllamaEmbeddingProvider
from app.rag.local_vector import LocalVectorKnowledgeIndex, VectorChunk


@dataclass(frozen=True)
class KnowledgeChunk:
    title: str
    content: str
    reference: str


class _HtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"h1", "h2", "h3", "p", "li", "br"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(part for part in self.parts if part.strip())


class KnowledgeBaseSearch:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        retrieval_mode: str | None = None,
        vector_store_dir: str | Path | None = None,
        knowledge_source: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.retrieval_mode = retrieval_mode
        self.vector_store_dir = Path(vector_store_dir) if vector_store_dir is not None else None
        self.knowledge_source = knowledge_source

    def search(self, query: str, limit: int = 3) -> list[KnowledgeChunk]:
        chunks = self._load_chunks()
        if not chunks:
            return []

        if self._knowledge_source() == "vector":
            return self._chroma_search(query, limit)

        mode = self._retrieval_mode()
        if mode == "keyword":
            return _keyword_search(query, chunks, limit)
        if mode == "vector":
            return self._vector_search(query, chunks, limit)
        return self._hybrid_search(query, chunks, limit)

    def _retrieval_mode(self) -> str:
        raw_mode = self.retrieval_mode or get_settings().knowledge_retrieval_mode
        mode = raw_mode.strip().lower()
        if mode in {"keyword", "vector", "hybrid"}:
            return mode
        return "hybrid"

    def _knowledge_source(self) -> str:
        settings = get_settings()
        raw_source = self.knowledge_source or settings.knowledge_source_default
        source = raw_source.strip().lower()
        if source in {"doc", "vector"}:
            return source
        return "doc"

    def _vector_store_dir(self) -> Path:
        if self.vector_store_dir is not None:
            return self.vector_store_dir
        return Path(get_settings().vector_store_dir)

    def _chroma_search(self, query: str, limit: int) -> list[KnowledgeChunk]:
        settings = get_settings()
        embedding_provider = OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            timeout_seconds=settings.ollama_embedding_timeout_seconds,
        )
        query_embedding = embedding_provider.embed_texts([query])[0]
        store = ChromaKnowledgeStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            min_score=settings.chroma_min_score,
        )
        hits = store.search(
            query_embedding=query_embedding,
            limit=min(max(limit, 1), max(settings.chroma_top_k, 1)),
        )
        return [
            KnowledgeChunk(
                title=hit.chunk.title,
                content=hit.chunk.content,
                reference=hit.chunk.reference,
            )
            for hit in hits
        ]

    def _vector_search(self, query: str, chunks: list[KnowledgeChunk], limit: int) -> list[KnowledgeChunk]:
        settings = get_settings()
        index = LocalVectorKnowledgeIndex(
            root_dir=self.root_dir,
            index_dir=self._vector_store_dir(),
        )
        vector_chunks = [_to_vector_chunk(chunk) for chunk in chunks]
        hits = index.search(
            query,
            vector_chunks,
            limit=limit,
            min_score=settings.knowledge_vector_min_score,
        )
        by_reference = {chunk.reference: chunk for chunk in chunks}
        return [by_reference[hit.chunk.reference] for hit in hits if hit.chunk.reference in by_reference]

    def _hybrid_search(self, query: str, chunks: list[KnowledgeChunk], limit: int) -> list[KnowledgeChunk]:
        combined: dict[str, tuple[float, KnowledgeChunk]] = {}

        keyword_ranked = _rank_keyword(query, chunks)
        max_keyword_score = max((score for score, _ in keyword_ranked), default=1)
        for score, chunk in keyword_ranked:
            if score < 2:
                continue
            _add_ranked_score(combined, chunk, score / max_keyword_score)

        settings = get_settings()
        index = LocalVectorKnowledgeIndex(root_dir=self.root_dir, index_dir=self._vector_store_dir())
        vector_hits = index.search(
            query,
            [_to_vector_chunk(chunk) for chunk in chunks],
            limit=max(limit * 2, 5),
            min_score=settings.knowledge_vector_min_score,
        )
        by_reference = {chunk.reference: chunk for chunk in chunks}
        for hit in vector_hits:
            chunk = by_reference.get(hit.chunk.reference)
            if chunk is not None:
                _add_ranked_score(combined, chunk, hit.score)

        ranked = sorted(combined.values(), key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:limit]]

    def _load_chunks(self) -> list[KnowledgeChunk]:
        if not self.root_dir.exists():
            return []

        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.root_dir.rglob("*")):
            if path.suffix.lower() == ".md":
                chunks.extend(_parse_markdown(path, self.root_dir))
            elif path.suffix.lower() in {".html", ".htm"}:
                chunks.extend(_parse_html(path, self.root_dir))
        return chunks

    def load_chunks(self) -> list[KnowledgeChunk]:
        return self._load_chunks()


def _keyword_search(query: str, chunks: list[KnowledgeChunk], limit: int) -> list[KnowledgeChunk]:
    ranked = [(score, chunk) for score, chunk in _rank_keyword(query, chunks) if score >= 2]
    return [chunk for _, chunk in ranked[:limit]]


def _rank_keyword(query: str, chunks: list[KnowledgeChunk]) -> list[tuple[int, KnowledgeChunk]]:
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    ranked: list[tuple[int, KnowledgeChunk]] = []
    for chunk in chunks:
        score = _score(query_terms, chunk)
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _add_ranked_score(
    combined: dict[str, tuple[float, KnowledgeChunk]],
    chunk: KnowledgeChunk,
    score: float,
) -> None:
    current_score = combined.get(chunk.reference, (0.0, chunk))[0]
    combined[chunk.reference] = (current_score + score, chunk)


def _to_vector_chunk(chunk: KnowledgeChunk) -> VectorChunk:
    return VectorChunk(title=chunk.title, content=chunk.content, reference=chunk.reference)


def _parse_markdown(path: Path, root_dir: Path) -> list[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^(#{1,3})\s*(.+)$", text)
    chunks: list[KnowledgeChunk] = []
    current_title = path.stem

    if sections[0].strip():
        chunks.append(_chunk(path, root_dir, current_title, sections[0]))

    for index in range(1, len(sections), 3):
        title = sections[index + 1].strip()
        content = sections[index + 2].strip() if index + 2 < len(sections) else ""
        current_title = title or current_title
        if content:
            chunks.append(_chunk(path, root_dir, current_title, content))
    return chunks


def _parse_html(path: Path, root_dir: Path) -> list[KnowledgeChunk]:
    parser = _HtmlTextParser()
    parser.feed(path.read_text(encoding="utf-8"))
    lines = [line.strip() for line in parser.text().splitlines() if line.strip()]

    chunks: list[KnowledgeChunk] = []
    current_title = path.stem
    current_content: list[str] = []

    for line in lines:
        if _looks_like_heading(line):
            if current_content:
                chunks.append(_chunk(path, root_dir, current_title, "\n".join(current_content)))
                current_content = []
            current_title = line
        else:
            current_content.append(line)

    if current_content:
        chunks.append(_chunk(path, root_dir, current_title, "\n".join(current_content)))
    return chunks


def _chunk(path: Path, root_dir: Path, title: str, content: str) -> KnowledgeChunk:
    relative_path = path.relative_to(root_dir).as_posix()
    slug = re.sub(r"\s+", "-", title.strip())
    return KnowledgeChunk(
        title=title.strip(),
        content=content.strip(),
        reference=f"{relative_path}#{slug}",
    )


def _tokenize(text: str) -> set[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    terms = {term for term in normalized.split() if term}
    for length in range(2, min(7, len(text) + 1)):
        terms.update(text[index : index + length] for index in range(0, len(text) - length + 1))
    return terms


def _score(query_terms: set[str], chunk: KnowledgeChunk) -> int:
    searchable = f"{chunk.title}\n{chunk.content}".lower()
    return sum(1 for term in query_terms if term in searchable)


def _looks_like_heading(line: str) -> bool:
    return len(line) <= 40 and not line.endswith(("。", "，", ".", ",", "；", ";"))
