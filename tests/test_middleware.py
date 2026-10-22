from fastapi.testclient import TestClient

from app.main import app


def test_health_contains_observability_headers():
    client = TestClient(app)
    response = client.get("/health", headers={"X-Request-ID": "test-request-001"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-001"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_health_generates_request_id_when_missing():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")