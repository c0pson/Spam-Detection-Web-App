from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False

from spam_detection.core.config import MODEL_DIR, MAX_LENGTH, DEFAULT_TOP_K


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", 
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "will", 
    "with", "you", "your", "this", "but", "they", "have", "had", "do", "does", "did",
    "not", "no", "yes", "can", "could", "should", "would", "may", "might", "must",
    "i", "me", "we", "us", "what", "which", "who", "when", "where", "why", "how"
}


def _merge_wordpieces(tokens: List[str], scores: List[float]) -> List[Tuple[str, float]]:
    merged: List[Tuple[str, float]] = []
    current_word = ""
    current_score = 0.0
    current_count = 0

    def flush() -> None:
        nonlocal current_word, current_score, current_count
        if current_word:
            merged.append((current_word, current_score / max(current_count, 1)))
        current_word = ""
        current_score = 0.0
        current_count = 0

    for tok, sc in zip(tokens, scores):
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            flush()
            continue

        if tok.startswith("##"):
            current_word += tok[2:]
        else:
            flush()
            current_word = tok

        current_score += float(sc)
        current_count += 1

    flush()
    # Filter out non-alphanumeric words, short words, and stop words
    merged = [
        (w, s) for (w, s) in merged 
        if re.search(r"[A-Za-z0-9]", w) and len(w) > 2 and w.lower() not in STOP_WORDS
    ]
    return merged


@dataclass(frozen=True)
class SpamPrediction:
    is_spam: bool
    spam_probability: float
    confidence: float
    keywords: Optional[List[Tuple[str, float]]] = None


class SpamChecker:
    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        max_length: int = MAX_LENGTH,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_dir = model_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> SpamPrediction:
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

        return SpamPrediction(
            is_spam=is_spam,
            spam_probability=spam_prob,
            confidence=confidence,
        )

    def explain(self, text: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Explainability requires 'captum'. Install it to enable keyword explanations.")

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
        return merged[:top_k]

    def analyze(self, text: str, explain: bool = False, top_k: int = DEFAULT_TOP_K) -> SpamPrediction:
        pred = self.predict(text)
        if not explain:
            return pred

        keywords = self.explain(text, top_k=top_k)
        return SpamPrediction(
            is_spam=pred.is_spam,
            spam_probability=pred.spam_probability,
            confidence=pred.confidence,
            keywords=keywords,
        )

class SpamService:
    _checker: Optional[SpamChecker] = None

    @classmethod
    def _get_checker(cls) -> SpamChecker:
        if cls._checker is None:
            cls._checker = SpamChecker()
        return cls._checker

    @classmethod
    def classify_text(cls, text: str, explain: bool = False, top_k: int = DEFAULT_TOP_K) -> SpamPrediction:
        checker = cls._get_checker()
        return checker.analyze(text, explain=explain, top_k=top_k)