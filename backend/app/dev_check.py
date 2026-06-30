from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import shutil
import sys


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class EnvironmentReport:
    status: CheckStatus
    python_version: str
    has_node: bool
    has_npm: bool
    has_backend_dir: bool
    has_frontend_dir: bool
    has_knowledge_base_dir: bool
    has_env_example: bool
    missing_items: list[str]


def check_local_environment(project_root: str | Path) -> EnvironmentReport:
    root = Path(project_root)
    checks = {
        "backend/": (root / "backend").is_dir(),
        "frontend/": (root / "frontend").is_dir(),
        "knowledge_base/": (root / "knowledge_base").is_dir(),
        ".env.example": (root / ".env.example").is_file(),
    }
    missing_items = [name for name, exists in checks.items() if not exists]

    return EnvironmentReport(
        status=CheckStatus.PASS if not missing_items else CheckStatus.FAIL,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        has_node=shutil.which("node") is not None,
        has_npm=shutil.which("npm") is not None,
        has_backend_dir=checks["backend/"],
        has_frontend_dir=checks["frontend/"],
        has_knowledge_base_dir=checks["knowledge_base/"],
        has_env_example=checks[".env.example"],
        missing_items=missing_items,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    report = check_local_environment(project_root)

    print(f"status: {report.status}")
    print(f"python: {report.python_version}")
    print(f"node: {'found' if report.has_node else 'missing'}")
    print(f"npm: {'found' if report.has_npm else 'missing'}")

    if report.missing_items:
        print("missing:")
        for item in report.missing_items:
            print(f"- {item}")
        return 1

    print("local environment paths are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
