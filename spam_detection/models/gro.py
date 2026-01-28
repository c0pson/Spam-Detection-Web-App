from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from spam_detection.core.logger import Logger

logger = Logger.get_logger(__name__)

@dataclass(frozen=True)
class GroWindowConfig:
    max_length: int = 256
    stride: int = 96

@dataclass(frozen=True)
class GroPrediction:
    is_spam: bool
    spam_probability: float
    confidence: float
    window_probabilities: list[float]

class GeneralRiskOverseer:
    """General Risk Overseer (G.R.O.) - BERT-based spam classifier.
    """
    SPAM_LABEL_INDEX = 1
    HAM_LABEL_INDEX = 0

    def __init__(self, model_dir: str, window: GroWindowConfig = GroWindowConfig(), device: str | None = None) -> None:
        self.window = window
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing GeneralRiskOverseer with device: {self.device}")
        logger.debug(f"Window config - max_length: {window.max_length}, stride: {window.stride}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("GeneralRiskOverseer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GeneralRiskOverseer: {e}", exc_info=True)
            raise

    @torch.no_grad()
    def window_spam_probabilities(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.window.max_length,
            stride=self.window.stride,
            return_overflowing_tokens=True,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)[:, self.SPAM_LABEL_INDEX]
        return probs.detach().cpu().tolist()

    def predict(self, text: str, threshold: float = 0.5) -> GroPrediction:
        logger.debug(f"Making GRO prediction for text (length: {len(text)}, threshold: {threshold})")
        try:
            window_probs = self.window_spam_probabilities(text)
            if not window_probs:
                logger.warning("No window probabilities generated, returning default prediction")
                return GroPrediction(is_spam=False, spam_probability=0.0, confidence=1.0, window_probabilities=[])
            prod_not_spam = 1.0
            for p in window_probs:
                prod_not_spam *= (1.0 - float(p))
            spam_prob = 1.0 - prod_not_spam
            is_spam = spam_prob >= threshold
            confidence = spam_prob if is_spam else 1.0 - spam_prob
            logger.debug(f"GRO Prediction: is_spam={is_spam}, spam_probability={spam_prob:.4f}, confidence={confidence:.4f}")
            return GroPrediction(
                is_spam=is_spam,
                spam_probability=spam_prob,
                confidence=confidence,
                window_probabilities=window_probs,
            )
        except Exception as e:
            logger.error(f"Error during GRO prediction: {e}", exc_info=True)
            raise
