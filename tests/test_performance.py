import pytest
import time
from spam_detection.main import create_app
from spam_detection.services.spam_checker import SpamService, SpamPrediction


@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.fixture(autouse=True)
def mock_spam_service(monkeypatch):
    """Mock SpamService to avoid real model inference."""

    def fake_classify(cls, text, explain=False, top_k=5):
        return SpamPrediction(
            is_spam=False,
            spam_probability=0.1,
            confidence=0.9,
            keywords=[("safe", 0.1)] if explain else [],
        )

    monkeypatch.setattr(SpamService, "classify_text", classmethod(fake_classify))


class TestPerformance:
    """Performance and benchmark tests."""

    def test_form_get_response_time_fast(self, client):
        """GET /form should respond quickly (< 100ms)."""
        start = time.time()
        resp = client.get("/form")
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 0.1, f"GET /form took {elapsed:.3f}s, expected < 0.1s"

    def test_form_post_response_time_fast(self, client):
        """POST /form with classification should respond quickly (< 500ms)."""
        start = time.time()
        resp = client.post("/form", data={"email_body": "Hello world"})
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 0.5, f"POST /form took {elapsed:.3f}s, expected < 0.5s"

    def test_form_post_large_input_performance(self, client):
        """POST /form with large input (50KB) should still respond < 1s."""
        large_text = "a" * 50000
        start = time.time()
        resp = client.post("/form", data={"email_body": large_text})
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 1.0, f"POST /form with 50KB took {elapsed:.3f}s, expected < 1.0s"

    def test_concurrent_requests_throughput(self, client):
        """Multiple sequential requests should maintain consistent response times."""
        times = []
        for i in range(10):
            start = time.time()
            resp = client.post("/form", data={"email_body": f"Email {i}"})
            elapsed = time.time() - start
            assert resp.status_code == 200
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        # Average response time should be < 200ms
        assert avg_time < 0.2, f"Average response time {avg_time:.3f}s, expected < 0.2s"
        # No individual request should take > 500ms
        assert all(t < 0.5 for t in times), f"Some requests slow: {times}"

    def test_form_response_size_reasonable(self, client):
        """Response HTML should be reasonably sized (< 10KB for not-spam)."""
        resp = client.post("/form", data={"email_body": "Hello"})
        assert resp.status_code == 200
        assert len(resp.data) < 10000, f"Response too large: {len(resp.data)} bytes"

    def test_get_vs_post_performance_ratio(self, client):
        """GET should be significantly faster than POST (at least 2x)."""
        # GET timing
        get_times = []
        for _ in range(5):
            start = time.time()
            client.get("/form")
            get_times.append(time.time() - start)

        # POST timing
        post_times = []
        for _ in range(5):
            start = time.time()
            client.post("/form", data={"email_body": "test"})
            post_times.append(time.time() - start)

        avg_get = sum(get_times) / len(get_times)
        avg_post = sum(post_times) / len(post_times)

        # POST should not be more than 3x slower
        ratio = avg_post / avg_get if avg_get > 0 else 1
        assert ratio < 3.0, f"POST is {ratio:.1f}x slower than GET, expected < 3x"

    def test_repeated_classification_consistent(self, client):
        """Repeated classification of same text should be consistent."""
        text = "Buy now for free!"
        responses = []

        for _ in range(5):
            resp = client.post("/form", data={"email_body": text})
            assert resp.status_code == 200
            responses.append(resp.data)

        # All responses should be identical (mocked service, deterministic)
        assert all(r == responses[0] for r in responses), "Responses vary unexpectedly"

    def test_empty_input_fast(self, client):
        """Empty input should be handled quickly."""
        start = time.time()
        resp = client.post("/form", data={"email_body": ""})
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 0.1, f"Empty input took {elapsed:.3f}s, expected < 0.1s"

    def test_unicode_input_performance(self, client):
        """Unicode input should not cause performance degradation."""
        unicode_text = "你好世界 🌍 مرحبا بالعالم" * 100
        start = time.time()
        resp = client.post("/form", data={"email_body": unicode_text})
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 0.3, f"Unicode input took {elapsed:.3f}s, expected < 0.3s"
