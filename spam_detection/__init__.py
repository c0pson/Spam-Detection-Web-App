"""Spam Detection Package - BERT-based email/message spam classifier.

This package provides spam detection and classification capabilities using
the General Risk Overseer (G.R.O.) model, a BERT-based sequence classifier.
"""

from spam_detection.core.config import (
    DEFAULT_TOP_K,
    MAX_LENGTH,
    MODEL_DIR,
    PROJECT_ROOT,
    STOP_WORDS,
)
from spam_detection.core.logger import Logger
from spam_detection.models.gro import (
    GeneralRiskOverseer,
    GroPrediction,
    GroWindowConfig,
)
from spam_detection.services.spam_checker import (
    SpamChecker,
    SpamPrediction,
)

__all__ = [
    # Configuration
    "DEFAULT_TOP_K",
    "MAX_LENGTH",
    "MODEL_DIR",
    "PROJECT_ROOT",
    "STOP_WORDS",
    # Logging
    "Logger",
    # Models
    "GeneralRiskOverseer",
    "GroPrediction",
    "GroWindowConfig",
    # Services
    "SpamChecker",
    "SpamPrediction",
]

__version__ = "1.0.0"
