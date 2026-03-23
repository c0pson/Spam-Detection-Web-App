import pytest
from requests.exceptions import ConnectionError

from spam_detection.services.spam_checker import SpamPrediction

@pytest.fixture
def client(monkeypatch):
    from spam_detection.main import create_app

    app = create_app()
    app.testing = True
    # prevent real sleeps in retry logic
    import spam_detection.api.routes_spam as routes_spam

    monkeypatch.setattr(routes_spam, "time", routes_spam.time)
    monkeypatch.setattr(routes_spam.time, "sleep", lambda _s: None)

    return app.test_client()

@pytest.fixture(autouse=True)
def default_mock(monkeypatch):
    """Default SpamService.classify_text mock (not spam)."""

    def fake_classify(cls, text, explain=False, top_k=5):
        return SpamPrediction(
            is_spam=False,
            spam_probability=0.05,
            confidence=0.9,
            keywords=[("safe", 0.1)] if explain else [],
        )

    monkeypatch.setattr("spam_detection.services.spam_checker.SpamService.classify_text", classmethod(fake_classify))

def test_get_form_shows_textarea(client):
    resp = client.get("/form")
    assert resp.status_code == 200
    assert b'name="email_body"' in resp.data

def test_post_form_not_spam_shows_not_spam_and_no_keywords(client):
    resp = client.post("/form", data={"email_body": "Hello there"})
    assert resp.status_code == 200
    assert b"Not Spam" in resp.data
    # keywords block should not be present for non-spam results
    assert b"Top keywords" not in resp.data

def test_post_form_spam_shows_keywords_and_user_text(client, monkeypatch):
    def spammy(cls, text, explain=False, top_k=5):
        return SpamPrediction(
            is_spam=True,
            spam_probability=0.99,
            confidence=0.99,
            keywords=[("free", 0.9), ("winner", 0.8)],
        )

    monkeypatch.setattr("spam_detection.services.spam_checker.SpamService.classify_text", classmethod(spammy))
    text = "You are a winner! Claim free prize"
    resp = client.post("/form", data={"email_body": text})
    assert resp.status_code == 200
    assert b"SPAM" in resp.data or b"spam" in resp.data
    # keywords should appear
    assert b"free" in resp.data
    assert b"winner" in resp.data
    # user text should be echoed
    assert text.encode("utf-8") in resp.data

def test_retry_on_transient_errors(client, monkeypatch):
    calls = {"count": 0}

    def flaky(cls, text, explain=False, top_k=5):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient")
        return SpamPrediction(is_spam=False, spam_probability=0.1, confidence=0.5, keywords=[])

    monkeypatch.setattr("spam_detection.services.spam_checker.SpamService.classify_text", classmethod(flaky))
    resp = client.post("/form", data={"email_body": "retry me"})
    assert resp.status_code == 200
    assert calls["count"] == 3
