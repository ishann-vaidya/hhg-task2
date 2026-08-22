"""Central configuration for the voice-enabled RAG pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

# ── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME = "ai4bharat/MSMARCO-XI"

# Supported languages (maps to per-language parquet files on Hugging Face)
AVAILABLE_LANGUAGES = [
    "as",  # Assamese
    "bn",  # Bengali
    "gu",  # Gujarati
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ne",  # Nepali
    "or",  # Odia
    "pa",  # Punjabi
    "sa",  # Sanskrit
    "ta",  # Tamil
    "te",  # Telugu
]

# HF repo stores one parquet per language, e.g. validation/hinval.parquet
LANGUAGE_FILE_PREFIX = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
}

# Primary language for the hackathon demo (Hindi has the widest tooling support)
DEFAULT_LANGUAGE = "hi"

# Use validation split — smaller than train, good for hackathon prototyping
DEFAULT_SPLIT = "validation"

# Subset size: full dataset is ~55 GB across all languages.
# 2 000 examples ≈ a few thousand unique passages — enough for a solid demo
# without multi-hour indexing. Increase to 5 000–10 000 for richer retrieval.
SUBSET_SIZE = 2_000

# ── Chunking (Phase 1) ───────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50

# Semantic chunking: split when similarity between adjacent sentences drops below this
SEMANTIC_SIMILARITY_THRESHOLD = 0.72
SEMANTIC_MAX_CHUNK_TOKENS = 384

# Embedding model used only for semantic chunking (local — no API key)
SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding model used for retrieval (multilingual for Indic languages)
RETRIEVAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

