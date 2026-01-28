"""Core configuration module.

Provides environment variables, constants, and logging infrastructure used
across the spam detection package.
"""

from spam_detection.core.config import (
    DEFAULT_TOP_K,
    ERROR_LOG_FILE,
    LOG_BACKUP_COUNT,
    LOG_DIR,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    MAX_LENGTH,
    MODEL_DIR,
    PROJECT_ROOT,
    STOP_WORDS,
)
from spam_detection.core.logger import Logger

__all__ = [
    # Configuration
    "DEFAULT_TOP_K",
    "MAX_LENGTH",
    "MODEL_DIR",
    "PROJECT_ROOT",
    "STOP_WORDS",
    # Logging
    "Logger",
    "LOG_DIR",
    "LOG_FILE",
    "ERROR_LOG_FILE",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_MAX_BYTES",
    "LOG_BACKUP_COUNT",
]
