"""Embedding generation and FAISS index construction for text chunks."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import INDEX_DIR, RETRIEVAL_EMBEDDING_MODEL
from src.chunking.base import Chunk
from src.chunking.pipeline import load_chunks


class ChunkIndexer:
    """Manages embedding generation, FAISS index construction, and serialization."""

    def __init__(self, embedding_model_name: str = RETRIEVAL_EMBEDDING_MODEL):
        self.model_name = embedding_model_name
        self.model = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_index(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
    ) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
        """Generate normalized embeddings and build a FAISS flat IP index (cosine similarity)."""
        if not chunks:
            raise ValueError("No chunks provided to index.")

        model = self._get_model()
        texts = [chunk.text for chunk in chunks]

        # Generate normalized embeddings (inner product corresponds to cosine similarity)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        metadata_list = [chunk.to_dict() for chunk in chunks]
        return index, metadata_list

    def save_index(
        self,
        index: faiss.IndexFlatIP,
        metadata_list: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Serialize FAISS index and metadata JSON to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = output_dir / "index.faiss"
        meta_path = output_dir / "metadata.json"

        faiss.write_index(index, str(faiss_path))
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    def load_index(self, input_dir: Path) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
        """Load FAISS index and metadata JSON from disk."""
        faiss_path = input_dir / "index.faiss"
        meta_path = input_dir / "metadata.json"

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"FAISS index or metadata not found in {input_dir}")

        index = faiss.read_index(str(faiss_path))
        with meta_path.open("r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        return index, metadata_list
