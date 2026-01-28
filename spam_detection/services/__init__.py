"""Services module.

Provides high-level APIs for spam detection and classification tasks.
"""

from spam_detection.services.spam_checker import (
    SpamChecker,
    SpamPrediction,
)

__all__ = [
    "SpamChecker",
    "SpamPrediction",
]
