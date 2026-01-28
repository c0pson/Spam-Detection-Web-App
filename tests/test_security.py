import pytest

from spam_detection.main import create_app
from spam_detection.services.spam_checker import SpamService, SpamPrediction


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def client():
    """Create Flask test client."""
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.fixture(autouse=True)
def mock_spam_service(monkeypatch):
    """Mock SpamService.classify_text to avoid model loading."""

    def fake_classify(cls, text, explain=False, top_k=5):
        return SpamPrediction(
            is_spam=False,
            spam_probability=0.1,
            confidence=0.9,
            keywords=[("safe", 0.1)] if explain else [],
        )

    monkeypatch.setattr(
        SpamService,
        "classify_text",
        classmethod(fake_classify),
    )


# -----------------------------
# XSS tests
# -----------------------------

class TestSecurityXSS:
    """XSS prevention and input sanitization tests."""

    @pytest.mark.parametrize(
        "payload,escaped_marker,forbidden_marker",
        [
            ("<script>alert(\"XSS\")</script>", b"&lt;script", b"<script>"),
            ("<img src=x onerror=alert(1)>", b"&lt;img", b"<img "),
            ("<div onclick=malicious()>", b"&lt;div", b"<div "),
        ],
    )
    def test_malicious_html_escaped(
        self,
        client,
        payload,
        escaped_marker,
        forbidden_marker,
    ):
        resp = client.post("/form", data={"email_body": payload})

        assert resp.status_code == 200
        # Ensure the exact submitted payload isn't injected unescaped, and escaped form appears
        assert payload.encode("utf-8") not in resp.data
        assert escaped_marker in resp.data


# -----------------------------
# Input validation tests
# -----------------------------

class TestSecurityInputValidation:
    """Input validation edge-case tests."""

    @pytest.mark.parametrize(
        "payload",
        [
            "",
            None,
            "   \n\t   ",
            "a" * 50000,
            "你好，世界",
            "مرحبا بالعالم",
            "😀😃😄😁",
            "!@#$%^&*()[]{}|<>?",
            "Hello\x00World",
            "' OR '1'='1",
            "; rm -rf /",
            "../../../etc/passwd",
        ],
        ids=[
            "empty",
            "none",
            "whitespace",
            "very_long",
            "chinese",
            "arabic",
            "emoji",
            "special_chars",
            "null_byte",
            "sql_like",
            "cmd_like",
            "path_traversal",
        ],
    )
    def test_various_inputs_do_not_crash(self, client, payload):
        data = {} if payload is None else {"email_body": payload}
        resp = client.post("/form", data=data)

        assert resp.status_code == 200
        assert b"Traceback (most recent call" not in resp.data
        assert b"Exception" not in resp.data

    def test_unicode_and_encoding(self, client):
        text = "测试 Arabic: العربية Emoji: 🔥"
        resp = client.post("/form", data={"email_body": text})

        assert resp.status_code == 200
        assert "utf-8" in resp.headers.get("Content-Type", "").lower()
        assert "测试".encode("utf-8") in resp.data
