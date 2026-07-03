from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


INDEX_VERSION = 1
INDEX_FILE_NAME = "knowledge_base_vector_index.json"
SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".html", ".htm"}


@dataclass(frozen=True)
class VectorChunk:
    title: str
    content: str
    reference: str


@dataclass(frozen=True)
class VectorSearchHit:
    chunk: VectorChunk
    score: float


class LocalVectorKnowledgeIndex:
    def __init__(self, *, root_dir: str | Path, index_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / INDEX_FILE_NAME

    def search(
        self,
        query: str,
        chunks: list[VectorChunk],
        *,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> list[VectorSearchHit]:
        query_vector = _vectorize(query)
        if not query_vector:
            return []

        entries = self._load_or_build(chunks)
        hits = [
            VectorSearchHit(
                chunk=entry.chunk,
                score=_cosine_similarity(query_vector, entry.vector),
            )
            for entry in entries
        ]
        hits = [hit for hit in hits if hit.score >= min_score]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _load_or_build(self, chunks: list[VectorChunk]) -> list["_VectorIndexEntry"]:
        signature = _source_signature(self.root_dir)
        existing = self._read_index()
        if existing is not None and existing.signature == signature:
            return existing.entries

        entries = [
            _VectorIndexEntry(
                chunk=chunk,
                vector=_vectorize(f"{chunk.title}\n{chunk.title}\n{chunk.content}"),
            )
            for chunk in chunks
        ]
        self._write_index(_VectorIndex(signature=signature, entries=entries))
        return entries

    def _read_index(self) -> "_VectorIndex | None":
        if not self.index_path.is_file():
            return None
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if payload.get("version") != INDEX_VERSION:
            return None

        raw_signature = payload.get("signature")
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_signature, list) or not isinstance(raw_chunks, list):
            return None

        entries: list[_VectorIndexEntry] = []
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict):
                return None
            chunk_payload = raw_chunk.get("chunk")
            vector_payload = raw_chunk.get("vector")
            if not isinstance(chunk_payload, dict) or not isinstance(vector_payload, dict):
                return None
            entries.append(
                _VectorIndexEntry(
                    chunk=VectorChunk(
                        title=str(chunk_payload.get("title", "")),
                        content=str(chunk_payload.get("content", "")),
                        reference=str(chunk_payload.get("reference", "")),
                    ),
                    vector={
                        str(key): float(value)
                        for key, value in vector_payload.items()
                        if isinstance(value, int | float)
                    },
                )
            )

        return _VectorIndex(
            signature=[str(item) for item in raw_signature],
            entries=entries,
        )

    def _write_index(self, index: "_VectorIndex") -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "signature": index.signature,
            "chunks": [
                {
                    "chunk": {
                        "title": entry.chunk.title,
                        "content": entry.chunk.content,
                        "reference": entry.chunk.reference,
                    },
                    "vector": entry.vector,
                }
                for entry in index.entries
            ],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class _VectorIndexEntry:
    chunk: VectorChunk
    vector: dict[str, float]


@dataclass(frozen=True)
class _VectorIndex:
    signature: list[str]
    entries: list[_VectorIndexEntry]


def _source_signature(root_dir: Path) -> list[str]:
    if not root_dir.exists():
        return []

    signature: list[str] = []
    for path in sorted(root_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES or not path.is_file():
            continue
        relative_path = path.relative_to(root_dir).as_posix()
        signature.append(f"{relative_path}:{_file_hash(path)}")
    return signature


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _vectorize(text: str) -> dict[str, float]:
    expanded = _expand_domain_terms(text.lower())
    counts: Counter[str] = Counter(_tokenize(expanded))
    return {term: float(count) for term, count in counts.items()}


def _expand_domain_terms(text: str) -> str:
    expansions = []
    rules = {
        "充钱": "充值",
        "充值": "充钱",
        "没到账": "未到账 不到账 到账",
        "不到账": "未到账 没到账 到账",
        "未到账": "不到账 没到账 到账",
        "封号": "封禁",
        "被封": "封禁",
        "申诉": "转人工 处理",
    }
    for keyword, expansion in rules.items():
        if keyword in text:
            expansions.append(expansion)
    if expansions:
        return f"{text} {' '.join(expansions)}"
    return text


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw_token):
            tokens.extend(_char_ngrams(raw_token))
        else:
            tokens.append(raw_token)
    return tokens


def _char_ngrams(text: str) -> list[str]:
    grams: list[str] = []
    max_length = min(6, len(text))
    for length in range(1, max_length + 1):
        grams.extend(text[index : index + length] for index in range(0, len(text) - length + 1))
    return grams


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))
