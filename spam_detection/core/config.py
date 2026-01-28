"""Configuration module for spam detection application.

This module defines application-wide configuration constants including:
- Model directory and parameters
- Logging configuration
- Text processing parameters (stop words, max length)
- Environment variable defaults

All configuration values can be overridden via environment variables.
"""

import os
from pathlib import Path

# Model Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = os.getenv(
    "SPAM_MODEL_DIR",
    str(PROJECT_ROOT / "models" / "gro_spam_v1"),
)
MAX_LENGTH = int(os.getenv("SPAM_MAX_LENGTH", "256"))
DEFAULT_TOP_K = int(os.getenv("SPAM_TOP_K", "5"))

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", 
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "will", 
    "with", "you", "your", "this", "but", "they", "have", "had", "do", "does", "did",
    "not", "no", "yes", "can", "could", "should", "would", "may", "might", "must",
    "i", "me", "we", "us", "what", "which", "who", "when", "where", "why", "how"
}

# Log Configuration
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "app.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
LOG_BACKUP_COUNT = 5
LOG_MAX_BYTES = 10485760
