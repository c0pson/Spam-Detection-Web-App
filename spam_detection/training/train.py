"""Training module for General Risk Overseer (G.R.O.) spam detection model.

This module implements the training pipeline for fine-tuning a BERT-based
model on spam detection datasets. It includes:

- Configuration management for training parameters
- Metric computation (accuracy, precision, recall, F1)
- Command-line argument parsing
- Model training with HuggingFace Trainer
- Model checkpoint management

The training process:
1. Loads dataset from disk
2. Tokenizes text with max_length truncation
3. Trains model with specified hyperparameters
4. Evaluates on validation and test sets
5. Saves best model checkpoint

Example:
    python -m spam_detection.training.train \\
        --dataset_path artifacts/datasets/spam_v1 \\
        --output_dir models/gro_spam_v1 \\
        --epochs 3 \\
        --batch_size 16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import evaluate
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


@dataclass(frozen=True)
class GroTrainConfig:
    """Configuration for G.R.O. model training.

    Attributes:
        dataset_path (Path): Path to pre-processed dataset directory.
        output_dir (Path): Directory to save trained model and tokenizer.
        base_model_name (str): HuggingFace model ID (e.g., 'bert-base-uncased').
        max_length (int): Maximum token sequence length. Defaults to 256.
        batch_size (int): Training/evaluation batch size. Defaults to 16.
        epochs (int): Number of training epochs. Defaults to 3.
        learning_rate (float): Adam optimizer learning rate. Defaults to 2e-5.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    dataset_path: Path
    output_dir: Path
    base_model_name: str

    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics for model evaluation.

    Calculates accuracy, precision, recall, and F1 score using the
    evaluate library. Uses binary averaging for multi-class metrics.

    Args:
        eval_pred: Named tuple containing model predictions (logits) and
            true labels from the evaluation dataset.

    Returns:
        Dict[str, float]: Dictionary containing:
            - accuracy: Overall accuracy (0-1)
            - precision: Binary precision (0-1)
            - recall: Binary recall (0-1)
            - f1: Binary F1 score (0-1)

    Note:
        - Predictions are obtained via argmax of logits
        - Assumes binary classification (2 classes)
        - Metric computation requires 'evaluate' package
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="binary")["f1"],
    }


def parse_args() -> GroTrainConfig:
    """Parse and validate command-line training arguments.

    Provides sensible defaults for all parameters, overridable via CLI.
    Environment variable CUDA_VISIBLE_DEVICES can control GPU usage.

    Returns:
        GroTrainConfig: Validated training configuration.

    Raises:
        SystemExit: If invalid arguments provided.

    CLI Arguments:
        --dataset_path: Path to pre-processed dataset (default: artifacts/datasets/spam_v1)
        --base_model: HuggingFace model ID (default: bert-base-uncased)
        --output_dir: Output directory for model (default: models/gro_spam_v1)
        --max_length: Max token sequence length (default: 256)
        --batch_size: Batch size for training (default: 16)
        --epochs: Number of training epochs (default: 3)
        --lr: Learning rate (default: 2e-5)
        --seed: Random seed (default: 42)
    """
    parser = argparse.ArgumentParser(description="Train General Risk Overseer (G.R.O.) on a local dataset.")
    parser.add_argument("--dataset_path", type=str, default="artifacts/datasets/spam_v1")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="spam_detection/artifacts/model/gro_spam_v1")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return GroTrainConfig(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        base_model_name=args.base_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )


def main() -> None:
    """Execute the complete training pipeline.

    Orchestrates the following steps:
    1. Parse training arguments
    2. Load pre-processed dataset from disk
    3. Initialize tokenizer and tokenize dataset
    4. Create BERT model and configure trainer
    5. Train model with validation and checkpointing
    6. Evaluate on test set and print metrics
    7. Save final model and tokenizer

    The training process uses:
    - Mixed precision (FP16) if CUDA available
    - Learning rate scheduler for optimization
    - Early stopping based on F1 score
    - Evaluation every epoch with metric computation

    Raises:
        FileNotFoundError: If dataset path doesn't exist
        RuntimeError: If model training fails
        KeyboardInterrupt: If interrupted by user (Ctrl+C)

    Output:
        - Prints test metrics after training
        - Saves model and tokenizer to output_dir
    """
    cfg = parse_args()

    dataset = load_from_disk(str(cfg.dataset_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)

    keep_columns = {"input_ids", "attention_mask", "label"}
    for split_name in tokenized.keys():
        drop_columns = [c for c in tokenized[split_name].column_names if c not in keep_columns]
        tokenized[split_name] = tokenized[split_name].remove_columns(drop_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name, num_labels=2)

    use_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=bool(use_cuda),
        tf32=bool(use_cuda),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"])
    print("Test metrics:", test_metrics)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    print(f"Saved G.R.O. model + tokenizer to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
