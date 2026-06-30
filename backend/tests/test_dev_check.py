from pathlib import Path

from app.dev_check import CheckStatus, check_local_environment


def test_check_local_environment_passes_for_existing_project_paths(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "backend").mkdir()
    (project_root / "frontend").mkdir()
    (project_root / "knowledge_base").mkdir()
    (project_root / ".env.example").write_text("APP_ENV=local\n", encoding="utf-8")

    report = check_local_environment(project_root)

    assert report.status == CheckStatus.PASS
    assert report.has_backend_dir is True
    assert report.has_frontend_dir is True
    assert report.has_knowledge_base_dir is True
    assert report.has_env_example is True


def test_check_local_environment_fails_when_required_paths_are_missing(tmp_path: Path) -> None:
    report = check_local_environment(tmp_path)

    assert report.status == CheckStatus.FAIL
    assert "backend/" in report.missing_items
    assert "frontend/" in report.missing_items
    assert "knowledge_base/" in report.missing_items
    assert ".env.example" in report.missing_items
