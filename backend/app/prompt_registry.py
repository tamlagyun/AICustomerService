from pathlib import Path

from app.config import Settings


class PromptNotFoundError(RuntimeError):
    pass


PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"
SUPPORTED_PROMPT_TYPES = {"decision", "planner", "followup_decision", "final_reply"}


def get_prompt_versions(settings: Settings) -> dict[str, str]:
    return {
        "decision": settings.prompt_decision_version,
        "planner": settings.prompt_planner_version,
        "followup_decision": settings.prompt_followup_decision_version,
        "final_reply": settings.prompt_final_reply_version,
    }


def load_prompt(prompt_type: str, version: str) -> str:
    if prompt_type not in SUPPORTED_PROMPT_TYPES:
        raise PromptNotFoundError(f"Unsupported prompt type: {prompt_type}")

    prompt_path = PROMPT_ROOT / prompt_type / f"{version}.txt"
    if not prompt_path.is_file():
        raise PromptNotFoundError(
            f"Prompt file not found for {prompt_type} version {version}: {prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8").strip()
