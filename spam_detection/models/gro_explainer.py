from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from spam_detection.models.gro import GroWindowConfig, GeneralRiskOverseer


@dataclass(frozen=True)
class GroExplanationConfig:
    top_k: int = 12
    ig_steps: int = 24


def _merge_wordpieces(tokens: List[str], scores: List[float]) -> List[Tuple[str, float]]:
    merged: List[Tuple[str, float]] = []
    current = ""
    current_score = 0.0
    current_count = 0

    def flush() -> None:
        nonlocal current, current_score, current_count
        if current:
            merged.append((current, current_score / max(current_count, 1)))
        current = ""
        current_score = 0.0
        current_count = 0

    for tok, sc in zip(tokens, scores):
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            flush()
            continue

        if tok.startswith("##"):
            current += tok[2:]
        else:
            flush()
            current = tok

        current_score += float(sc)
        current_count += 1

    flush()
    return [(w, s) for (w, s) in merged if re.search(r"[A-Za-z0-9]", w)]


class GroExplainer:

    def __init__(self, model_dir: str, window: GroWindowConfig = GroWindowConfig(), device: str | None = None) -> None:
        self.window = window
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def explain_keywords(self, text: str, cfg: GroExplanationConfig = GroExplanationConfig()) -> List[Tuple[str, float]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.window.max_length,
            stride=self.window.stride,
            return_overflowing_tokens=True,
            padding=True,
        )

        input_ids_all = encoded["input_ids"]
        attention_mask_all = encoded["attention_mask"]

        embeddings = self.model.get_input_embeddings()

        def forward_with_embeds(input_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=mask)
            return outputs.logits[:, GeneralRiskOverseer.SPAM_LABEL_INDEX]  # spam logit

        ig = IntegratedGradients(forward_with_embeds)
        global_scores: Dict[str, float] = defaultdict(float)

        for i in range(input_ids_all.size(0)):
            input_ids = input_ids_all[i : i + 1].to(self.device)
            attention_mask = attention_mask_all[i : i + 1].to(self.device)

            baseline_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)

            input_embeds = embeddings(input_ids)
            baseline_embeds = embeddings(baseline_ids)

            attributions = ig.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                n_steps=cfg.ig_steps,
            )

            token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
            merged = _merge_wordpieces(tokens, token_scores)

            for word, score in merged:
                if score > 0:
                    global_scores[word.lower()] += float(score)

        ranked = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: cfg.top_k]
