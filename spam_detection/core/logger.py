"""Logger module for spam detection application.

Provides a professional Logger class for managing application logging with
file rotation, error separation, and console output.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from spam_detection.core.config import (
    ERROR_LOG_FILE,
    LOG_BACKUP_COUNT,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_MAX_BYTES,
)

class Logger:
    """Professional logger with file rotation, error separation, and console output.

    This class provides a centralized logging interface with:
    - Rotating file handler to prevent disk space issues
    - Separate error log file for critical issues
    - Console output for development/debugging
    - Configurable log levels and formats

    Example:
        >>> logger = Logger.get_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
    """

    _loggers: dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the specified name.

        Args:
            name: The name of the logger (typically __name__).

        Returns:
            A configured logging.Logger instance.
        """
        if name in cls._loggers:
            return cls._loggers[name]
        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
        if not logger.handlers:
            cls._setup_handlers(logger)
        cls._loggers[name] = logger
        return logger

    @classmethod
    def _setup_handlers(cls, logger: logging.Logger) -> None:
        """Set up all handlers for the logger.

        Args:
            logger: The logger instance to configure.
        """
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        error_handler = RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the logger cache. Useful for testing."""
        cls._loggers.clear()
