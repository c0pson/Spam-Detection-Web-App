import pytest
from spam_detection.models.gro import GeneralRiskOverseer

def make_gro_with_window_probs(probs):
    gro = GeneralRiskOverseer.__new__(GeneralRiskOverseer)
    def window_spam_probabilities(self, text):
        return probs
    gro.window_spam_probabilities = window_spam_probabilities.__get__(gro, GeneralRiskOverseer)
    return gro

def test_gro_predict_calculation():
    gro = make_gro_with_window_probs([0.2, 0.3])
    pred = gro.predict("dummy text")
    prod_not_spam = (1 - 0.2) * (1 - 0.3)
    expected_spam = 1 - prod_not_spam
    assert pred.spam_probability == pytest.approx(expected_spam)
    assert pred.is_spam == (expected_spam >= 0.5)

def test_gro_predict_empty_windows():
    gro = make_gro_with_window_probs([])
    pred = gro.predict("dummy")
    assert pred.is_spam is False
    assert pred.spam_probability == 0.0
    assert pred.window_probabilities == []

def test_gro_predict_threshold_equal():
    gro = make_gro_with_window_probs([0.5])
    pred = gro.predict("dummy", threshold=0.5)
    assert pred.spam_probability == pytest.approx(0.5)
    assert pred.is_spam is True
