from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMUsageRecord:
    provider: str
    model: str
    operation: str
    usage: LLMTokenUsage
    estimated_cost: float = 0.0


@dataclass
class LLMUsageSummary:
    records: list[LLMUsageRecord] = field(default_factory=list)

    def add(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        usage: LLMTokenUsage,
        input_token_price_per_1k: float = 0.0,
        output_token_price_per_1k: float = 0.0,
    ) -> None:
        self.records.append(
            LLMUsageRecord(
                provider=provider,
                model=model,
                operation=operation,
                usage=usage,
                estimated_cost=estimate_llm_cost(
                    usage,
                    input_token_price_per_1k=input_token_price_per_1k,
                    output_token_price_per_1k=output_token_price_per_1k,
                ),
            )
        )

    def to_audit_payload(self) -> dict[str, object]:
        prompt_tokens = sum(record.usage.prompt_tokens for record in self.records)
        completion_tokens = sum(record.usage.completion_tokens for record in self.records)
        total_tokens = sum(record.usage.total_tokens for record in self.records)
        estimated_cost = sum(record.estimated_cost for record in self.records)
        return {
            "llm_prompt_tokens": prompt_tokens,
            "llm_completion_tokens": completion_tokens,
            "llm_total_tokens": total_tokens,
            "llm_estimated_cost": estimated_cost,
        }


def parse_llm_token_usage(payload: object) -> LLMTokenUsage | None:
    if not isinstance(payload, dict):
        return None

    prompt_tokens = _int_value(payload.get("prompt_tokens"))
    completion_tokens = _int_value(payload.get("completion_tokens"))
    total_tokens = _int_value(payload.get("total_tokens"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    total_tokens = total_tokens or prompt_tokens + completion_tokens
    return LLMTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def estimate_llm_cost(
    usage: LLMTokenUsage,
    *,
    input_token_price_per_1k: float = 0.0,
    output_token_price_per_1k: float = 0.0,
) -> float:
    return (
        usage.prompt_tokens / 1000 * input_token_price_per_1k
        + usage.completion_tokens / 1000 * output_token_price_per_1k
    )


def empty_llm_usage_audit_payload() -> dict[str, object]:
    return {
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "llm_estimated_cost": 0.0,
    }


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    return None
