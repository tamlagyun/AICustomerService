from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass(frozen=True)
class StructuredOutputRepairResult:
    success: bool
    content: str = ""
    repaired: bool = False
    reason: str = ""


def repair_structured_json(raw_content: str) -> StructuredOutputRepairResult:
    original = raw_content.strip()
    if not original:
        return StructuredOutputRepairResult(success=False, reason="empty content")

    candidates = _candidate_json_texts(original)
    for candidate in candidates:
        normalized = _normalize_candidate(candidate)
        parsed = _parse_json_like(normalized)
        if parsed.success:
            return StructuredOutputRepairResult(
                success=True,
                content=_dump_compact_json(parsed.value),
                repaired=parsed.source != original,
            )

    return StructuredOutputRepairResult(
        success=False,
        reason="unable to extract valid JSON object or array",
    )


@dataclass(frozen=True)
class _ParsedCandidate:
    success: bool
    value: Any = None
    source: str = ""


def _candidate_json_texts(raw_content: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_markdown_json_blocks(raw_content))
    extracted = _extract_balanced_json(raw_content)
    if extracted:
        candidates.append(extracted)
    candidates.append(raw_content)
    wrapped = _wrap_missing_outer_object(raw_content)
    if wrapped:
        candidates.append(wrapped)
    return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))


def _markdown_json_blocks(raw_content: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(raw_content):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks


def _extract_balanced_json(raw_content: str) -> str:
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = raw_content.find(open_char)
        if start < 0:
            continue
        extracted = _balanced_slice(raw_content, start, open_char, close_char)
        if extracted:
            return extracted
    return ""


def _balanced_slice(raw_content: str, start: int, open_char: str, close_char: str) -> str:
    depth = 0
    in_string = False
    quote = ""
    escaped = False

    for index in range(start, len(raw_content)):
        char = raw_content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == open_char:
            depth += 1
            continue
        if char == close_char:
            depth -= 1
            if depth == 0:
                return raw_content[start : index + 1]
    return ""


def _wrap_missing_outer_object(raw_content: str) -> str:
    stripped = raw_content.strip()
    if stripped.startswith(("{", "[")):
        return ""
    if ":" not in stripped:
        return ""
    return "{" + stripped.strip().strip(",") + "}"


def _normalize_candidate(candidate: str) -> str:
    normalized = candidate.strip()
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = _remove_trailing_commas(normalized)
    return normalized


def _remove_trailing_commas(candidate: str) -> str:
    previous = candidate
    while True:
        current = re.sub(r",\s*([}\]])", r"\1", previous)
        if current == previous:
            return current
        previous = current


def _parse_json_like(candidate: str) -> _ParsedCandidate:
    try:
        return _ParsedCandidate(success=True, value=json.loads(candidate), source=candidate)
    except json.JSONDecodeError:
        pass

    try:
        value = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return _ParsedCandidate(success=False)

    if isinstance(value, dict | list):
        return _ParsedCandidate(success=True, value=value, source=candidate)
    return _ParsedCandidate(success=False)


def _dump_compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
