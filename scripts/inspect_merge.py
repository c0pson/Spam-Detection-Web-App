from spam_detection.services import spam_checker


def main():
    tokens = ["[CLS]", "he", "##llo", "world", "[SEP]"]
    scores = [0.0, 1.0, 1.0, 2.0, 0.0]
    merged = spam_checker._merge_wordpieces(tokens, scores)
    print("Merged:", merged)


if __name__ == "__main__":
    main()
