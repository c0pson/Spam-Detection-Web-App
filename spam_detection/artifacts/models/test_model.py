from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os


MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "models/bert-enron-spam"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
SEED = 42


def load_enron_dataset():

    print("[INFO] Loading SetFit/enron_spam...")
    raw_datasets = load_dataset("SetFit/enron_spam")

    full_train = raw_datasets["train"]
    test_ds = raw_datasets["test"]

    split = full_train.train_test_split(test_size=0.1, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]

    def add_labels(example):
        example["labels"] = example["label"]
        return example

    train_ds = train_ds.map(add_labels)
    val_ds = val_ds.map(add_labels)
    test_ds = test_ds.map(add_labels)

    return train_ds, val_ds, test_ds


def tokenize_datasets(train_ds, val_ds, test_ds, tokenizer):

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    print("[INFO] Tokenizing dataset...")
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    return train_ds, val_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dataset
    train_ds, val_ds, test_ds = load_enron_dataset()

    # 2. Tokenizer
    print(f"[INFO] Loading: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds, val_ds, test_ds = tokenize_datasets(train_ds, val_ds, test_ds, tokenizer)

    # 3. Model
    print(f"[INFO] Loading: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # 0 = ham, 1 = spam
    )

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=SEED,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Training
    print("[INFO] Training starting...")
    trainer.train()

    print("[INFO] Dataset evaluation...")
    test_metrics = trainer.evaluate(test_ds)
    print("[RESULTS] Test metrics:", test_metrics)

    # 7. Saving
    print(f"[INFO] Model saved to {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[INFO] Ready.")


if __name__ == "__main__":
    main()
