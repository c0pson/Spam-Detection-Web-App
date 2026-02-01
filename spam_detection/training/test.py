"""Evaluation module for the G.R.O. spam detection model.

Runs inference on a dataset split and prints standard classification metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from spam_detection.core.config import MAX_LENGTH, MODEL_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _pick_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

DEFAULT_DATASET_PATH = _pick_existing_path(
    [
        PROJECT_ROOT / "spam_detection" / "artifacts" / "datasets" / "spam_v1",
        PROJECT_ROOT / "artifacts" / "datasets" / "spam_v1",
    ]
)
DEFAULT_MODEL_DIR = _pick_existing_path(
    [
        PROJECT_ROOT / "model" / "gro_spam_v1",
        PROJECT_ROOT / "models" / "gro_spam_v1",
        Path(MODEL_DIR),
    ]
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spam_detection" / "artifacts" / "eval"


@dataclass(frozen=True)
class GroTestConfig:
    dataset_path: Path
    model_dir: Path
    output_dir: Path
    max_length: int
    batch_size: int
    split: str
    threshold: float
    seed: int
    quiet: bool


def _parse_args() -> GroTestConfig:
    parser = argparse.ArgumentParser(description="Evaluate a trained G.R.O. model.")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Do not print metrics to stdout.")
    args = parser.parse_args()
    return GroTestConfig(
        dataset_path=Path(args.dataset_path),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        batch_size=args.batch_size,
        split=args.split,
        threshold=args.threshold,
        seed=args.seed,
        quiet=args.quiet,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _compute_binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "npv": npv,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _tokenize_split(dataset, tokenizer, max_length: int):
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    keep_columns = {"input_ids", "attention_mask", "label"}
    drop_columns = [c for c in tokenized.column_names if c not in keep_columns]
    if drop_columns:
        tokenized = tokenized.remove_columns(drop_columns)
    return tokenized


def _write_metrics(metrics: Dict[str, float], cfg: GroTestConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": cfg.split,
        "threshold": cfg.threshold,
        "metrics": metrics,
    }
    out_path = cfg.output_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    txt_path = cfg.output_dir / "metrics.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Split: {cfg.split}\n")
        handle.write(f"Threshold: {cfg.threshold:.3f}\n")
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if key in {"tp", "tn", "fp", "fn"}:
                handle.write(f"{key}: {int(value)}\n")
            else:
                handle.write(f"{key}: {value:.6f}\n")
        handle.write("\nExplanations:\n")
        handle.write("accuracy: (TP + TN) / (TP + TN + FP + FN)\n")
        handle.write("precision: TP / (TP + FP)\n")
        handle.write("recall: TP / (TP + FN)\n")
        handle.write("f1: harmonic mean of precision and recall\n")
        handle.write("specificity: TN / (TN + FP)\n")
        handle.write("fpr: FP / (FP + TN)\n")
        handle.write("fnr: FN / (FN + TP)\n")
        handle.write("npv: TN / (TN + FN)\n")
        handle.write("tp: true positives (spam predicted as spam)\n")
        handle.write("tn: true negatives (ham predicted as ham)\n")
        handle.write("fp: false positives (ham predicted as spam)\n")
        handle.write("fn: false negatives (spam predicted as ham)\n")
    return out_path


def main() -> None:
    cfg = _parse_args()

    dataset = load_from_disk(str(cfg.dataset_path))
    if cfg.split not in dataset:
        raise KeyError(f"Split '{cfg.split}' not found in dataset. Available: {list(dataset.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(cfg.model_dir))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_cuda = torch.cuda.is_available()
    eval_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        seed=cfg.seed,
        per_device_eval_batch_size=cfg.batch_size,
        fp16=bool(use_cuda),
        tf32=bool(use_cuda),
        report_to="none",
    )

    split_dataset = _tokenize_split(dataset[cfg.split], tokenizer, cfg.max_length)
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=split_dataset,
        data_collator=data_collator,
    )

    pred_output = trainer.predict(split_dataset)
    logits = np.asarray(pred_output.predictions)
    labels = np.asarray(pred_output.label_ids)
    probs = _softmax(logits)[:, 1]
    metrics = _compute_binary_metrics(probs, labels, cfg.threshold)

    _write_metrics(metrics, cfg)

    if not cfg.quiet:
        print(f"Split: {cfg.split}")
        print(f"Threshold: {cfg.threshold:.3f}")
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if key in {"tp", "tn", "fp", "fn"}:
                print(f"{key}: {int(value)}")
            else:
                print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
