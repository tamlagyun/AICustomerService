from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re


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
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def search(self, query: str, limit: int = 3) -> list[KnowledgeChunk]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        ranked: list[tuple[int, KnowledgeChunk]] = []
        for chunk in self._load_chunks():
            score = _score(query_terms, chunk)
            if score >= 2:
                ranked.append((score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
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
