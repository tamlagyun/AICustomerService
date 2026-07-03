import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Settings

_MANAGED_HANDLER_ATTR = "_customer_service_agent_file_handler"


def configure_logging(settings: Settings) -> Path:
    log_dir = resolve_log_dir(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(_log_level(settings.log_level))
    _remove_stale_managed_handlers(root_logger, log_file)

    if not _has_managed_handler(root_logger, log_file):
        handler = RotatingFileHandler(
            log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        setattr(handler, _MANAGED_HANDLER_ATTR, True)
        handler.setLevel(_log_level(settings.log_level))
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)

    return log_file


def resolve_log_dir(log_dir: str) -> Path:
    path = Path(log_dir)
    if path.is_absolute():
        return path
    backend_dir = Path(__file__).resolve().parents[1]
    return (backend_dir / path).resolve()


def _log_level(level_name: str) -> int:
    level = getattr(logging, level_name.strip().upper(), logging.INFO)
    return level if isinstance(level, int) else logging.INFO


def _has_managed_handler(logger: logging.Logger, log_file: Path) -> bool:
    expected = str(log_file)
    for handler in logger.handlers:
        if getattr(handler, _MANAGED_HANDLER_ATTR, False) and getattr(
            handler,
            "baseFilename",
            None,
        ) == expected:
            return True
    return False


def _remove_stale_managed_handlers(logger: logging.Logger, log_file: Path) -> None:
    expected = str(log_file)
    for handler in list(logger.handlers):
        if not getattr(handler, _MANAGED_HANDLER_ATTR, False):
            continue
        if getattr(handler, "baseFilename", None) == expected:
            continue
        logger.removeHandler(handler)
        handler.close()
