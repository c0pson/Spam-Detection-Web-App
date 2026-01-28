import pytest
from spam_detection.main import create_app
from spam_detection.services.spam_checker import SpamPrediction, SpamService

def test_home_get():
    app = create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.data

def test_form_get():
    app = create_app()
    client = app.test_client()
    resp = client.get("/form")
    assert resp.status_code == 200

def test_form_post_with_mocked_service(monkeypatch):
    def fake_classify_text(cls, text, explain=False, top_k=5):
        return SpamPrediction(is_spam=True, spam_probability=0.9, confidence=0.9, keywords=[("spammy", 0.9)])

    monkeypatch.setattr(SpamService, "classify_text", classmethod(fake_classify_text))
    app = create_app()
    client = app.test_client()
    resp = client.post("/form", data={"email_body": "Buy now!"})
    assert resp.status_code == 200
    assert b"Buy now!" in resp.data
