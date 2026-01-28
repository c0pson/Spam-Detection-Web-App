import pytest
from spam_detection.main import create_app
from spam_detection.api import routes_spam
from spam_detection.services import spam_checker as sc_module
from spam_detection.services.spam_checker import SpamPrediction
import requests

def test_spamservice_singleton(monkeypatch):
    created = []

    class FakeChecker:
        def __init__(self):
            created.append(object())
    monkeypatch.setattr(sc_module, "SpamChecker", FakeChecker)
    sc_module.SpamService._checker = None
    c1 = sc_module.SpamService._get_checker()
    c2 = sc_module.SpamService._get_checker()
    assert c1 is c2
    assert len(created) == 1

def test_show_form_retries_and_success(monkeypatch):
    calls = {"n": 0}

    class FakeService:
        @classmethod
        def classify_text(cls, text, explain=False, top_k=5):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.exceptions.ConnectionError("transient")
            return SpamPrediction(is_spam=True, spam_probability=0.9, confidence=0.9, keywords=[("spammy", 0.9)])

    monkeypatch.setattr(routes_spam, "SpamService", FakeService)
    monkeypatch.setattr(routes_spam, "time", type("T", (), {"sleep": lambda s: None}))
    app = create_app()
    client = app.test_client()
    resp = client.post("/form", data={"email_body": "Retry test"})
    assert resp.status_code == 200
    assert b"Retry test" in resp.data
    assert calls["n"] == 3
