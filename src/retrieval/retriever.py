"""Vector database retrieval using FAISS index."""

import time
from pathlib import Path
from typing import Any

from config.settings import INDEX_DIR
from src.indexing.indexer import ChunkIndexer


class VectorRetriever:
    """Handles query encoding, FAISS similarity search, and metadata mapping."""

    def __init__(
        self,
        strategy: str = "metadata_aware",
        language: str = "hi",
        index_dir: Path | None = None,
    ):
        self.strategy = strategy
        self.language = language
        self.index_dir = index_dir or (INDEX_DIR / strategy / language)
        self.indexer = ChunkIndexer()
        self.index = None
        self.metadata = None

    def _ensure_loaded(self) -> None:
        """Lazy load the FAISS index and metadata."""
        if self.index is None or self.metadata is None:
            self.index, self.metadata = self.indexer.load_index(self.index_dir)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Encode the query, search the FAISS index, and return top-k matching chunks with similarity scores."""
        self._ensure_loaded()
        if not query.strip():
            return []

        # Time the retrieval process
        start_time = time.perf_counter()

        model = self.indexer._get_model()
        # Encode query and normalize for cosine similarity via inner product
        query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

        # Search index
        scores, indices = self.index.search(query_vector, top_k)
        retrieval_ms = (time.perf_counter() - start_time) * 1000

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["similarity_score"] = float(score)
            meta["retrieval_latency_ms"] = retrieval_ms
            results.append(meta)

        return results
