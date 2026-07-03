import logging

from app.config import Settings
from app.logging_config import configure_logging


def test_configure_logging_writes_runtime_log_file(tmp_path) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        log_level="INFO",
        log_max_bytes=1024,
        log_backup_count=1,
    )

    log_file = configure_logging(settings)
    logging.getLogger("app.tests.logging_config").info("runtime log persisted")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file == tmp_path / "app.log"
    assert log_file.is_file()
    assert "runtime log persisted" in log_file.read_text(encoding="utf-8")

