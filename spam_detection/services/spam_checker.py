from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

from spam_detection.core.config import STOP_WORDS
from spam_detection.core.config import MODEL_DIR, MAX_LENGTH, DEFAULT_TOP_K
from spam_detection.core.logger import Logger

logger = Logger.get_logger(__name__)

def _flush(word: str, score: float, count: int) -> tuple[str, float] | None:
    if not word or count == 0:
        return None
    return word, score / count

def _merge_wordpieces(tokens: list[str], scores: list[float]) -> list[tuple[str, float]]:
    merged: list[tuple[str, float]] = []
    current_word = ""
    current_score = 0.0
    current_count = 0
    for tok, sc in zip(tokens, scores):
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            flushed = _flush(current_word, current_score, current_count)
            if flushed:
                merged.append(flushed)
            current_word = ""
            current_score = 0.0
            current_count = 0
            continue
        if tok.startswith("##"):
            current_word += tok[2:]
        else:
            flushed = _flush(current_word, current_score, current_count)
            if flushed:
                merged.append(flushed)
            current_word = tok
        current_score += float(sc)
        current_count += 1
    flushed = _flush(current_word, current_score, current_count)
    if flushed:
        merged.append(flushed)
    merged = [
        (w, s)
        for w, s in merged
        if re.search(r"[A-Za-z0-9]", w)
        and len(w) > 2
        and w.lower() not in STOP_WORDS
    ]
    return merged

@dataclass(frozen=True)
class SpamPrediction:
    is_spam: bool
    spam_probability: float
    confidence: float
    keywords: list[tuple[str, float]] | None = None

class SpamChecker:
    def __init__(self, model_dir: str = MODEL_DIR, max_length: int = MAX_LENGTH, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.max_length = max_length
        logger.info(f"Initializing SpamChecker with device: {device}")
        logger.debug(f"Model directory: {model_dir}, Max length: {max_length}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info("SpamChecker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpamChecker: {e}", exc_info=True)
            raise

    @torch.no_grad()
    def predict(self, text: str) -> SpamPrediction:
        logger.debug(f"Making prediction for text (length: {len(text)})")
        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            spam_prob = float(probs[1].cpu())
            is_spam = spam_prob >= 0.5
            confidence = spam_prob if is_spam else 1.0 - spam_prob
            logger.debug(f"Prediction: is_spam={is_spam}, spam_probability={spam_prob:.4f}")
            return SpamPrediction(
                is_spam=is_spam,
                spam_probability=spam_prob,
                confidence=confidence,
            )
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise

    def explain(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[str, float]]:
        logger.debug(f"Generating explanations for text (top_k={top_k})")
        try:
            from captum.attr import IntegratedGradients #type: ignore
            CAPTUM_AVAILABLE = True
        except Exception:
            logger.warning("Captum library not available for explanations")
            CAPTUM_AVAILABLE = False
        if not CAPTUM_AVAILABLE:
            logger.error("Explainability requires 'captum' library")
            raise RuntimeError("Explainability requires 'captum'. Install it to enable keyword explanations.")
        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            baseline_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
            embeddings = self.model.get_input_embeddings()

            def forward_with_embeds(input_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                outputs = self.model(inputs_embeds=input_embeds, attention_mask=mask)
                return outputs.logits[:, 1]

            ig = IntegratedGradients(forward_with_embeds)
            input_embeds = embeddings(input_ids)
            baseline_embeds = embeddings(baseline_ids)
            attributions = ig.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                n_steps=24,
            )
            token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
            merged = _merge_wordpieces(tokens, token_scores)
            merged.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Generated {len(merged[:top_k])} keyword explanations")
            return merged[:top_k]
        except Exception as e:
            logger.error(f"Error during explanation generation: {e}", exc_info=True)
            raise

    def analyze(self, text: str, explain: bool = False, top_k: int = DEFAULT_TOP_K) -> SpamPrediction:
        logger.info(f"Starting analysis for text (length: {len(text)}, explain={explain})")
        try:
            pred = self.predict(text)
            if not explain:
                return pred

            keywords = self.explain(text, top_k=top_k)
            result = SpamPrediction(
                is_spam=pred.is_spam,
                spam_probability=pred.spam_probability,
                confidence=pred.confidence,
                keywords=keywords,
            )
            logger.info(f"Analysis complete: is_spam={result.is_spam}, keywords={len(keywords)}")
            return result
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            raise

class SpamService:
    _checker: SpamChecker | None = None

    @classmethod
    def _get_checker(cls) -> SpamChecker:
        if cls._checker is None:
            logger.info("Initializing SpamService checker instance")
            cls._checker = SpamChecker()
        return cls._checker

    @classmethod
    def classify_text(cls, text: str, explain: bool = False, top_k: int = DEFAULT_TOP_K) -> SpamPrediction:
        checker = cls._get_checker()
        return checker.analyze(text, explain=explain, top_k=top_k)
