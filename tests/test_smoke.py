"""Smoke tests for the FastAPI application.

These tests exercise the app's HTTP surface only; they do not require
Ollama or any other external service.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"]
    assert body["version"]


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["health"] == "/health"
    assert body["docs"] == "/docs"


def test_expected_routes_registered():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = set(response.json()["paths"])
    assert "/health" in paths
    assert any(path.startswith("/agents") for path in paths)
    assert any(path.startswith("/rag") for path in paths)
    assert any(path.startswith("/memory") for path in paths)
