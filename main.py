"""FastAPI Backend Service for Voice-Enabled RAG pipeline."""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

# Set C-extension thread counts to 1 BEFORE importing numpy, faiss, or torch
# This prevents OpenMP / MKL / BLAS from spawning 16-thread stacks (16 * 64MB = 1GB RAM OOM on Railway 512MB RAM)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import VoiceRAGPipeline
from config.settings import INDEX_DIR

app = FastAPI(
    title="Indic Voice RAG API Service",
    description="Backend API for transcribing Hindi audio, performing vector search, and generating grounded responses.",
    version="1.0.0",
)

# Enable CORS for local development and deployed frontends.
frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
allowed_origins = [
    "http://localhost:5173",  # default Vite React port
    "http://localhost:3000",  # standard React port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory for audio files
TEMP_DIR = Path("data/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Log index directory state on startup for Railway diagnostics
try:
    logger.info("=== STARTUP: INDEX_DIR = %s ===", INDEX_DIR)
    logger.info("=== STARTUP: INDEX_DIR exists = %s ===", INDEX_DIR.exists())
except Exception as e:
    logger.warning("Startup index log notice: %s", e)


class TextQueryRequest(BaseModel):
    query: str
    strategy: str = "metadata_aware"
    language: str = "hi"
    threshold: float = 0.42
    mock: bool = False
    mock_answer: str | None = None


@app.get("/")
@app.get("/health")
def root_health() -> dict[str, Any]:
    """Root health check endpoint for Railway container readiness probes."""
    return {"status": "ok", "service": "Indic Voice RAG API", "version": "1.0.0"}


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Check API configuration status (are API keys loaded on the backend?)."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    return {
        "groq_configured": bool(groq_key),
        "sarvam_configured": bool(sarvam_key),
        "live_mode_ready": bool(groq_key and sarvam_key),
    }


@app.get("/api/debug")
def get_debug_info() -> dict[str, Any]:
    """Diagnostic endpoint — lists index files found on the server filesystem."""
    import sys
    cwd = Path.cwd()
    index_files: list[dict] = []
    if INDEX_DIR.exists():
        for p in sorted(INDEX_DIR.rglob("*")):
            index_files.append({
                "path": str(p),
                "is_file": p.is_file(),
                "size_bytes": p.stat().st_size if p.is_file() else None,
            })
    return {
        "cwd": str(cwd),
        "index_dir": str(INDEX_DIR),
        "index_dir_exists": INDEX_DIR.exists(),
        "index_files": index_files,
        "python_version": sys.version,
        "data_dir_contents": [str(p) for p in Path("data").iterdir()] if Path("data").exists() else [],
    }


@app.post("/api/predict/text")
def predict_text(request: TextQueryRequest) -> dict[str, Any]:
    """Execute the RAG pipeline starting directly from a text query (bypassing STT)."""
    try:
        pipeline = VoiceRAGPipeline(
            strategy=request.strategy,
            language=request.language,
            off_topic_threshold=request.threshold,
        )

        res = pipeline.run_pipeline(
            query_text=request.query,
            mock_stt=request.mock,
            mock_gen=request.mock,
            mock_gen_answer=request.mock_answer,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/api/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    strategy: str = Form("metadata_aware"),
    language: str = Form("hi"),
    threshold: float = Form(0.42),
    mock: bool = Form(False),
    mock_text: str = Form("निगम क्या है?"),
) -> dict[str, Any]:
    """Upload a spoken audio file, transcribe it via STT, and run the RAG query pipeline."""
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".wav", ".mp3", ".webm", ".ogg", ".m4a"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{file_ext}'. Upload WAV, MP3, WebM, or OGG.",
        )

    # Save incoming stream temporarily
    temp_path = TEMP_DIR / f"upload_{int(time.time())}{file_ext}"
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize and run pipeline
        pipeline = VoiceRAGPipeline(
            strategy=strategy,
            language=language,
            off_topic_threshold=threshold,
        )

        res = pipeline.run_pipeline(
            audio_path=temp_path,
            mock_stt=mock,
            mock_gen=mock,
            mock_stt_text=mock_text,
        )
        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    finally:
        # Guarantee cleanup of temporary uploaded audio file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.get("/api/latency")
def get_latency_report() -> dict[str, Any]:
    """Retrieve the latency analytics report generated by phase 7 benchmarking."""
    report_path = Path("data/latency_report.json")
    if not report_path.exists():
        return {
            "status": "warning",
            "message": "No latency benchmark data available. Please run scripts/phase7_latency_analytics.py first.",
            "summary": {},
        }

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read latency report: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

