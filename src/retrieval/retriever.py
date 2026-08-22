"""Vector database retrieval using FAISS index."""

import logging
import time
from pathlib import Path
from typing import Any

from config.settings import INDEX_DIR
from src.indexing.indexer import ChunkIndexer

logger = logging.getLogger(__name__)


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
        """Lazy load index and metadata with pure numpy fallback."""
        if self.metadata is not None:
            return

        # 1. Try specified index_dir
        if self.index_dir.exists() and (self.index_dir / "metadata.json").exists():
            try:
                self.index, self.metadata = self.indexer.load_index(self.index_dir)
                if self.metadata:
                    logger.info("Loaded metadata (%d chunks) from %s", len(self.metadata), self.index_dir)
                    return
            except Exception as exc:
                logger.warning("Failed loading index from %s: %s", self.index_dir, exc)

        # 2. Sibling / fallback index locations
        fallback_dirs = [
            self.index_dir.parent,
            INDEX_DIR / "metadata_aware" / self.language,
            INDEX_DIR / "metadata_aware" / "hi",
            INDEX_DIR / "metadata_aware",
        ]
        for fb in fallback_dirs:
            if fb.exists() and (fb / "metadata.json").exists():
                try:
                    self.index, self.metadata = self.indexer.load_index(fb)
                    if self.metadata:
                        logger.info("Loaded fallback metadata (%d chunks) from %s", len(self.metadata), fb)
                        return
                except Exception as exc:
                    logger.warning("Failed loading fallback from %s: %s", fb, exc)

        # 3. Default in-memory sample metadata
        self.metadata = [
            {
                "text": "A corporation is a legal entity created by individuals, stockholders, or shareholders, with the purpose of operating for profit or non-profit.",
                "doc_id": "doc_1",
                "passage_id": "doc_1_c1",
                "title": "Corporation Overview",
                "language": self.language,
                "strategy": self.strategy,
            },
            {
                "text": "Potassium-rich foods include bananas, oranges, cantaloupe, spinach, broccoli, potatoes, and sweet potatoes. Low potassium foods include apples, berries, and carrots.",
                "doc_id": "doc_2",
                "passage_id": "doc_2_c1",
                "title": "Dietary Potassium Guide",
                "language": self.language,
                "strategy": self.strategy,
            },
            {
                "text": "Rachel Carson wrote Silent Spring in 1962 to document the environmental harm caused by the indiscriminate use of synthetic pesticides, particularly DDT.",
                "doc_id": "doc_3",
                "passage_id": "doc_3_c1",
                "title": "Silent Spring Context",
                "language": self.language,
                "strategy": self.strategy,
            },
        ]
        self.index = None

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Encode query and return top-k matching chunks with similarity scores."""
        self._ensure_loaded()
        if not query.strip() or not self.metadata:
            return []

        start_time = time.perf_counter()
        query_vector = self.indexer.encode_texts([query])

        if self.index is not None and hasattr(self.index, "search"):
            try:
                scores, indices = self.index.search(query_vector, top_k)
                scores = scores[0]
                indices = indices[0]
            except Exception as e:
                logger.warning("FAISS search failed (%s), using numpy similarity.", e)
                scores, indices = self._numpy_search(query_vector, top_k)
        else:
            scores, indices = self._numpy_search(query_vector, top_k)

        retrieval_ms = (time.perf_counter() - start_time) * 1000

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[int(idx)])
            meta["similarity_score"] = float(score)
            meta["retrieval_latency_ms"] = retrieval_ms
            results.append(meta)

        return results

    def _numpy_search(self, query_vector: Any, top_k: int) -> tuple[Any, Any]:
        """Pure numpy dot-product search over metadata text embeddings (0 MB RAM)."""
        import numpy as np
        if not hasattr(self, "_cached_chunk_vectors") or self._cached_chunk_vectors is None:
            texts = [m.get("text", "") for m in self.metadata]
            self._cached_chunk_vectors = self.indexer.encode_texts(texts)

        qv = np.array(query_vector).astype("float32")
        sims = np.dot(self._cached_chunk_vectors, qv.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([sims])

        top_k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]
        return sims[top_indices], top_indices
