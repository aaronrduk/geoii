"""
Logging configuration for production-ready logging.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up production-grade logging configuration.

    Args:
        log_dir: Directory to save log files. If None, uses 'logs' directory
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        logging.Logger: Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger("svamitva")
    logger.setLevel(numeric_level)

    logger.handlers = []

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
            maxBytes=10 * 1024 * 1024,
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

    logger.info("Logging configured successfully")

    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("This is a log message")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(f"svamitva.{self.__class__.__name__}")
        return self._logger
