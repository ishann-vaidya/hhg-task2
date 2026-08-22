"""End-to-End Voice-Enabled RAG pipeline."""

import time
from pathlib import Path
from typing import Any

from src.generation.generator import GenerationInput, RAGOrchestrator
from src.guardrails.guard import RAGGuardrails
from src.retrieval.retriever import VectorRetriever
from src.stt.sarvam import SarvamSTT


class VoiceRAGPipeline:
    """Orchestrates STT, Retrieval, Generation, and Guardrails with detailed latency tracking."""

    def __init__(
        self,
        strategy: str = "metadata_aware",
        language: str = "hi",
        off_topic_threshold: float = 0.42,
        groq_model: str = "groq/compound-mini",
        stt_model: str = "saaras:v3",
    ):
        self.language = language
        self.stt = SarvamSTT(model=stt_model)
        self.retriever = VectorRetriever(strategy=strategy, language=language)
        self.generator = RAGOrchestrator(model=groq_model, language=language)
        self.guardrails = RAGGuardrails(off_topic_threshold=off_topic_threshold, language=language)

    def run_pipeline(
        self,
        audio_path: Path | str | None = None,
        query_text: str | None = None,
        mock_stt: bool = False,
        mock_gen: bool = False,
        mock_stt_text: str = "निगम क्या है?",
        mock_gen_answer: str | None = None,
    ) -> dict[str, Any]:
        """Run the full E2E pipeline, either starting from audio or direct text, tracking step latencies."""
        start_time = time.perf_counter()
        latencies = {}

        # 1. Speech-to-Text (if audio provided)
        if audio_path is not None:
            stt_start = time.perf_counter()
            try:
                # Map simple language codes to BCP-47 for Sarvam STT
                locale_map = {
                    "hi": "hi-IN",
                    "en": "en-IN",
                    "mr": "mr-IN",
                    "ta": "ta-IN",
                    "te": "te-IN"
                }
                stt_lang = locale_map.get(self.language, "hi-IN")

                stt_res = self.stt.transcribe(
                    audio_path=audio_path,
                    language_code=stt_lang,
                    mock=mock_stt,
                    mock_text=mock_stt_text,
                )
                query = stt_res.get("transcript", "")
                latencies["stt"] = (time.perf_counter() - stt_start) * 1000
            except Exception as e:
                return {
                    "status": "error",
                    "error_message": f"STT failed: {str(e)}",
                    "latencies": {"stt": (time.perf_counter() - stt_start) * 1000},
                }
        elif query_text is not None:
            query = query_text
            latencies["stt"] = 0.0  # text input bypasses STT
        else:
            raise ValueError("Either audio_path or query_text must be provided.")

        # 2. Safety Guardrail
        safety_start = time.perf_counter()
        safety_res = self.guardrails.verify_query(query)
        latencies["safety_guard"] = (time.perf_counter() - safety_start) * 1000

        if not safety_res["safe"]:
            total_ms = (time.perf_counter() - start_time) * 1000
            latencies["total"] = total_ms
            return {
                "status": "blocked_safety",
                "query": query,
                "answer": safety_res["reason"],
                "chunks": [],
                "latencies": latencies,
                "guardrails": {"safety": safety_res},
            }

        # 3. Retrieval
        retrieval_start = time.perf_counter()
        chunks = self.retriever.retrieve(query, top_k=3)
        latencies["retrieval"] = (time.perf_counter() - retrieval_start) * 1000

        # 4. Off-topic Guardrail
        off_topic_start = time.perf_counter()
        off_topic_res = self.guardrails.verify_retrieval(query, chunks)
        latencies["off_topic_guard"] = (time.perf_counter() - off_topic_start) * 1000

        if not off_topic_res["on_topic"]:
            total_ms = (time.perf_counter() - start_time) * 1000
            latencies["total"] = total_ms
            return {
                "status": "blocked_off_topic",
                "query": query,
                "answer": off_topic_res["reason"],
                "chunks": chunks,
                "latencies": latencies,
                "guardrails": {"safety": safety_res, "off_topic": off_topic_res},
            }

        # 5. Generation
        gen_start = time.perf_counter()
        gen_input = GenerationInput(query=query, chunks=chunks)
        gen_res = self.generator.generate_answer(
            gen_input,
            mock=mock_gen,
            mock_answer=mock_gen_answer,
        )
        latencies["generation"] = (time.perf_counter() - gen_start) * 1000

        # 6. Groundedness Guardrail
        grounded_start = time.perf_counter()
        grounded_res = self.guardrails.verify_groundedness(
            gen_res.answer,
            chunks,
            gen_res.grounded,
        )
        latencies["groundedness_guard"] = (time.perf_counter() - grounded_start) * 1000

        total_ms = (time.perf_counter() - start_time) * 1000
        latencies["total"] = total_ms

        status = "success" if grounded_res["grounded"] else "blocked_groundedness"
        final_answer = (
            gen_res.answer if grounded_res["grounded"] else grounded_res["reason"]
        )

        return {
            "status": status,
            "query": query,
            "answer": final_answer,
            "chunks": chunks,
            "latencies": latencies,
            "guardrails": {
                "safety": safety_res,
                "off_topic": off_topic_res,
                "groundedness": grounded_res,
            },
            "reasoning": gen_res.reasoning,
            "citations": gen_res.citations,
            "retries": gen_res.retries,
        }
