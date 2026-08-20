"""Speech-to-Text translation and transcription using the Sarvam API."""

import os
import time
from pathlib import Path
from typing import Any

import httpx


class SarvamSTT:
    """Wrapper for the Sarvam AI Speech-to-Text REST API."""

    def __init__(self, api_key: str | None = None, model: str = "saaras:v3"):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        self.model = model
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    def transcribe(
        self,
        audio_path: Path | str,
        language_code: str = "hi-IN",
        mock: bool = False,
        mock_text: str = "निगम क्या है?",
    ) -> dict[str, Any]:
        """Transcribe an audio file using Sarvam STT REST API.

        If mock is True or self.api_key is not set, a simulated mock response is returned.
        """
        start_time = time.perf_counter()

        if mock or not self.api_key:
            # Simulate network latency (e.g. 150 ms) for mock mode accuracy in benchmarks
            time.sleep(0.150)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "transcript": mock_text,
                "language_code": language_code,
                "model": self.model,
                "latency_ms": latency_ms,
                "status": "success",
                "mocked": True,
            }

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        headers = {"api-subscription-key": self.api_key}

        import mimetypes
        content_type, _ = mimetypes.guess_type(str(audio_path))
        content_type = content_type or "audio/wav"

        # Open in binary mode for multipart upload
        with audio_path.open("rb") as f:
            files = {"file": (audio_path.name, f.read(), content_type)}

        data = {"model": self.model}

        response = httpx.post(
            self.api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=30.0,
        )

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Sarvam API error {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        result = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "transcript": result.get("transcript", "").strip(),
            "language_code": language_code,
            "model": self.model,
            "latency_ms": latency_ms,
            "status": "success",
            "mocked": False,
        }
