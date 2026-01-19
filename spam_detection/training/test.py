from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from spam_detection.models.gro import GeneralRiskOverseer, GroWindowConfig
from spam_detection.models.gro_explainer import GroExplainer, GroExplanationConfig

END_MARKER = "END"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "can", "com", "could", "did", "do", "does", "doing", "down",
    "for", "from", "had", "has", "have", "having", "he", "hello", "her", "hers",
    "him", "his", "how", "https", "i", "if", "in", "into", "is", "it", "its",
    "just", "me", "more", "most", "my", "no", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "out", "over", "own",
    "same", "she", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "why", "will",
    "with", "within", "would", "you", "your", "yours",
}


def read_multiline_message(first_line: str) -> str:
    lines: List[str] = [first_line]
    print(f"[ Input ] Enter message body. Finish with '{END_MARKER}' on its own line.")

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == END_MARKER:
            break

        lines.append(line)

    return "\n".join(lines).strip()



def filter_keywords(keywords: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
    filtered: List[Tuple[str, float]] = []
    for word, score in keywords:
        w = word.strip().lower()
        if len(w) < 3:
            continue
        if w in STOPWORDS:
            continue
        filtered.append((word, score))
        if len(filtered) >= top_k:
            break
    return filtered


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "spam_detection" / "artifacts" / "model" / "gro_spam_v1"

    window_cfg = GroWindowConfig(max_length=256, stride=96)
    gro = GeneralRiskOverseer(model_dir=str(model_dir), window=window_cfg)
    explainer = GroExplainer(model_dir=str(model_dir), window=window_cfg)

    print("General Risk Overseer (G.R.O.) – Spam Classification CLI")
    print(f"Paste/type an email (multi-line). Finish with a single line: {END_MARKER}")
    print("Type 'exit' or 'quit' as the first line to stop.\n")

    while True:
        first_line = input("Email (first line) > ").strip()
        if first_line.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        print(f"(Continue input. Finish with a single line: {END_MARKER})")
        text = read_multiline_message(first_line)
        if not text:
            print("Empty input, try again.\n")
            continue

        print("[G.R.O. - BERT] Classifying...")

        pred = gro.predict(text, threshold=0.5)

        print("\n=== Verdict ===")
        print(f"Label           : {'SPAM' if pred.is_spam else 'HAM'}")
        print(f"Spam probability: {pred.spam_probability:.4f}\n")

        if pred.is_spam:
            ans = input("Explain why SPAM? (y/n) > ").strip().lower()
            if ans in {"y", "yes"}:
                print("[G.R.O. - BERT] Computing explanation (this may take a while for long emails)...")
                raw_keywords = explainer.explain_keywords(
                    text,
                    GroExplanationConfig(top_k=50, ig_steps=12),
                )
                keywords = filter_keywords(raw_keywords, top_k=12)

                print("\n=== Keywords ===")
                for word, score in keywords:
                    print(f"{word:20s} {score:.6f}")
                print()

if __name__ == "__main__":
    main()
