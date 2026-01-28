import pytest
from spam_detection.services import spam_checker

def test_flush():
    assert spam_checker._flush("", 1.0, 1) is None
    assert spam_checker._flush("hello", 3.0, 3) == ("hello", 1.0)
    assert spam_checker._flush("x", 1.0, 0) is None

def test_merge_wordpieces_basic():
    tokens = ["[CLS]", "he", "##llo", "world", "[SEP]"]
    scores = [0.0, 1.0, 1.0, 2.0, 0.0]
    merged = spam_checker._merge_wordpieces(tokens, scores)
    assert ("hello", pytest.approx(1.0)) in merged
    assert ("world", pytest.approx(4.0 / 3.0)) in merged

def test_merge_filters_stop_words_and_short_tokens():
    tokens = ["[CLS]", "an", "##d", "ok", "no", "abc", "[SEP]"]
    scores = [0.0, 0.5, 0.5, 0.2, 0.1, 1.0, 0.0]
    merged = spam_checker._merge_wordpieces(tokens, scores)
    assert all(w.lower() not in {"and", "ok"} for w, _ in merged)
    assert ("abc", pytest.approx(2.3 / 5.0)) in merged
