from __future__ import annotations

from enum import Enum
import os
from pathlib import Path

class SERVER(Enum):
    URL = os.getenv("SPAM_SERVER_URL", "http://79.76.44.20")
    CHECK_SPAM = "/check_spam"
    CHECK_AVAILABILITY = "/check_availability"
    AVAILABLE = 200

    def __add__(self, other: "SERVER") -> str:
        """Return the concatenation of two SERVER enum member values.

        Args:
            other (SERVER): Another SERVER enum member to add.

        Returns:
            str: The combined value of the two enum members.
        """
        return str(self.value) + str(other.value)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = os.getenv(
    "SPAM_MODEL_DIR",
    str(PROJECT_ROOT / "spam_detection" / "artifacts" / "model" / "gro_spam_v1"),
)
MAX_LENGTH = int(os.getenv("SPAM_MAX_LENGTH", "256"))
DEFAULT_TOP_K = int(os.getenv("SPAM_TOP_K", "5"))