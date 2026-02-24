"""
Logging configuration for production-ready logging.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys

# Global flag to prevent double initialization
_LOGGING_INITIALIZED = False


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up production-grade logging configuration.
    Safe against multiple calls.
    """
    global _LOGGING_INITIALIZED

    logger = logging.getLogger("svamitva")

    # Fast return if already initialized
    if _LOGGING_INITIALIZED:
        return logger

    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger.setLevel(numeric_level)

    # Prevent duplicate logs propagating to root logger
    logger.propagate = False

    # Do NOT clear handlers globally on the root logger.
    # Only clear handlers on this specific logger if it somehow has them before init
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "svamitva.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "svamitva_errors.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    _LOGGING_INITIALIZED = True
    logger.info("Logging configured successfully")

    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(f"svamitva.{self.__class__.__name__}")
        return self._logger
