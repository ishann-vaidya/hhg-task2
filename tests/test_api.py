"""Backend integration tests for FastAPI endpoints."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_api_status() -> None:
    """Verify status route responds with key configurations."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "groq_configured" in data
    assert "sarvam_configured" in data
    assert "live_mode_ready" in data


def test_api_latency_report() -> None:
    """Verify latency route responds with report json or config warn."""
    response = client.get("/api/latency")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "summary" in data


def test_api_predict_text_mock() -> None:
    """Verify text pipeline queries execute successfully in mock mode."""
    payload = {
        "query": "निगम क्या है?",
        "strategy": "metadata_aware",
        "threshold": 0.42,
        "mock": True,
        "mock_answer": "निगम एक कानूनी संगठन है।",
    }
    response = client.post("/api/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == "निगम क्या है?"
    assert data["answer"] == "निगम एक कानूनी संगठन है।"
    assert "latencies" in data
    assert data["latencies"]["stt"] == 0.0


def test_api_predict_audio_mock() -> None:
    """Verify audio pipeline queries execute successfully in mock mode."""
    files = {"file": ("test.wav", b"RIFFxxxxWAVEfmt xxxxdataxxxx", "audio/wav")}
    data = {
        "strategy": "metadata_aware",
        "threshold": 0.42,
        "mock": True,
        "mock_text": "निगम क्या है?",
    }
    response = client.post("/api/predict/audio", files=files, data=data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == "निगम क्या है?"
    assert "latencies" in data
    assert data["latencies"]["stt"] > 0.0
