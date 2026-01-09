from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 256
END_MARKER = "END"

def read_multiline_message() -> str:
    print(f"\nPaste/type your email. Finish with a single line: {END_MARKER}")
    lines: List[str] = []
    while True:
        line = input()
        if line.strip() == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    model_dir = repo_root / "spam_detection" / "artifacts" / "model" / "spam_v1"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Repo root : {repo_root}")
    print(f"Model dir : {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()

    print("\nType 'exit' or 'quit' as the FIRST line to stop.\n")

    while True:
        first_line = input("Email (first line) > ").rstrip("\n")

        if first_line.strip().lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        lines = [first_line]
        print(f"(Continue pasting/typing. Finish with a single line: {END_MARKER})")
        while True:
            line = input()
            if line.strip() == END_MARKER:
                break
            lines.append(line)

        text = "\n".join(lines).strip()
        if not text:
            print("Empty input, try again.\n")
            continue

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        spam_prob = float(probs[1].cpu())
        ham_prob = float(probs[0].cpu())

        print("\n=== Prediction ===")
        print(f"Label : {'SPAM' if spam_prob >= 0.5 else 'HAM'}")
        print(f"Spam probability: {spam_prob:.4f}")
        print(f"Ham probability : {ham_prob:.4f}\n")


if __name__ == "__main__":
    main()
