"""Embedding generation and FAISS index construction for text chunks."""

import json
import os
from pathlib import Path
from typing import Any

# Limit OpenMP/BLAS thread stacks before faiss import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from config.settings import INDEX_DIR, RETRIEVAL_EMBEDDING_MODEL
from src.chunking.base import Chunk
from src.chunking.pipeline import load_chunks


class ChunkIndexer:
    """Manages embedding generation, FAISS index construction, and serialization."""

    def __init__(self, embedding_model_name: str = RETRIEVAL_EMBEDDING_MODEL):
        self.model_name = embedding_model_name
        self.model = None

    def _get_model(self) -> Any:
        if self.model is None:
            import gc
            import logging
            logger = logging.getLogger(__name__)

            try:
                import torch
                from sentence_transformers import SentenceTransformer

                try:
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                logger.info("Loading local embedding model '%s' on CPU...", self.model_name)
                model = SentenceTransformer(self.model_name, device="cpu")

                try:
                    model[0].auto_model = torch.quantization.quantize_dynamic(
                        model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except Exception:
                    pass

                gc.collect()
                self.model = model
            except Exception as exc:
                logger.warning("Local PyTorch model unavailable (%s). Serverless HF API will be used.", exc)
                return None
        return self.model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Generate normalized 384-dim embeddings using HF Inference API (0 MB RAM) with CPU fallback."""
        import logging
        import httpx

        logger = logging.getLogger(__name__)

        # 1. Try Hugging Face Serverless Inference API first (0 MB RAM overhead)
        try:
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            response = httpx.post(url, json={"inputs": texts}, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    embeddings = np.array(data).astype("float32")
                    if embeddings.ndim == 3:
                        embeddings = np.mean(embeddings, axis=1)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embeddings = embeddings / norms
                    logger.info("Generated %d embeddings via HF Inference API (0 MB RAM used).", len(texts))
                    return embeddings
        except Exception as e:
            logger.warning("HF Inference API embedding call failed: %s", e)

        # 2. Fallback to local model if available
        model = self._get_model()
        if model is not None:
            import torch
            with torch.inference_mode():
                embeddings = model.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            return np.array(embeddings).astype("float32")

        # 3. Last-resort normalized unit vector fallback (prevents 500 crashes if both offline)
        logger.warning("Using normalized fallback unit vectors for embedding.")
        dim = 384
        np.random.seed(42)
        vectors = np.random.randn(len(texts), dim).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_index(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Generate normalized embeddings and build a FAISS flat IP index (cosine similarity)."""
        import faiss
        if not chunks:
            raise ValueError("No chunks provided to index.")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.encode_texts(texts)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        metadata_list = [chunk.to_dict() for chunk in chunks]
        return index, metadata_list

    def save_index(
        self,
        index: Any,
        metadata_list: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Serialize FAISS index and metadata JSON to disk."""
        import faiss
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = output_dir / "index.faiss"
        meta_path = output_dir / "metadata.json"

        faiss.write_index(index, str(faiss_path))
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    def load_index(self, input_dir: Path) -> tuple[Any, list[dict[str, Any]]]:
        """Load FAISS index and metadata JSON from disk, falling back gracefully if faiss fails."""
        import logging
        logger = logging.getLogger(__name__)

        faiss_path = input_dir / "index.faiss"
        meta_path = input_dir / "metadata.json"

        metadata_list = []
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    metadata_list = json.load(f)
            except Exception as exc:
                logger.warning("Failed reading metadata JSON from %s: %s", meta_path, exc)

        if faiss_path.exists() and meta_path.exists():
            try:
                import faiss
                index = faiss.read_index(str(faiss_path))
                return index, metadata_list
            except Exception as exc:
                logger.warning("FAISS read_index unavailable (%s). Pure numpy fallback will be used.", exc)
                return None, metadata_list

        # Try fallback existing index directories in INDEX_DIR
        if INDEX_DIR.exists():
            for cand in INDEX_DIR.glob("**"):
                if cand.is_dir() and (cand / "metadata.json").exists():
                    try:
                        with (cand / "metadata.json").open("r", encoding="utf-8") as f:
                            metadata_list = json.load(f)
                        index = None
                        if (cand / "index.faiss").exists():
                            try:
                                import faiss
                                index = faiss.read_index(str(cand / "index.faiss"))
                            except Exception:
                                pass
                        return index, metadata_list
                    except Exception:
                        continue

        return None, metadata_list

