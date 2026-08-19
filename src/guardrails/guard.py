"""Guardrails for safety, off-topic queries, and output groundedness verification."""

import re
from typing import Any


class RAGGuardrails:
    """Implements safety filters, off-topic detection, and hallucination checks."""

    def __init__(self, off_topic_threshold: float = 0.42):
        self.off_topic_threshold = off_topic_threshold
        # Heuristic unsafe phrase pattern for demonstration (English & Hindi)
        self.unsafe_pattern = re.compile(
            r"\b(बम|हत्या|गाली|suicide|kill|murder|bomb|abuse|fuck|terrorist|आतंकवादी)\b",
            re.IGNORECASE,
        )

    def verify_query(self, query: str) -> dict[str, Any]:
        """Screen query for safety violations."""
        if self.unsafe_pattern.search(query):
            return {
                "safe": False,
                "reason": "गाली-गलौज या असुरक्षित भाषा पाई गई। (Query contains unsafe or inappropriate language.)",
            }
        return {"safe": True, "reason": "Query is safe."}

    def verify_retrieval(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Detect off-topic queries by checking retrieval similarity scores against a threshold."""
        if not retrieved_chunks:
            return {
                "on_topic": False,
                "reason": "कोई प्रासंगिक जानकारी नहीं मिली। (No relevant context found. Query is off-topic.)",
            }

        # Check similarity of the best retrieved chunk
        best_score = retrieved_chunks[0].get("similarity_score", 0.0)
        if best_score < self.off_topic_threshold:
            return {
                "on_topic": False,
                "reason": (
                    f"यह प्रश्न हमारी जानकारी के दायरे से बाहर है (Similarity: {best_score:.4f} < {self.off_topic_threshold}). "
                    "(Query is off-topic.)"
                ),
            }

        return {"on_topic": True, "reason": "Query is on-topic."}

    def verify_groundedness(
        self,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
        llm_grounded: bool,
    ) -> dict[str, Any]:
        """Verify the generated answer is grounded in retrieved chunks to prevent hallucination."""
        # Trust LLM groundedness assessment
        if not llm_grounded:
            return {
                "grounded": False,
                "reason": "प्रदान किए गए संदर्भ में उत्तर समर्थित नहीं है। (LLM flagged response as not grounded.)",
            }

        # Heuristic word overlap verification to catch completely ungrounded text
        def clean_text(text: str) -> set[str]:
            words = re.findall(r"\w+", text.lower())
            # Basic Hindi stop words
            hindi_stops = {
                "है",
                "हैं",
                "का",
                "की",
                "के",
                "में",
                "से",
                "को",
                "और",
                "यह",
                "वह",
                "जो",
                "तो",
                "पर",
                "हो",
                "भी",
                "एक",
            }
            return {w for w in words if w not in hindi_stops and len(w) > 1}

        answer_words = clean_text(answer)
        context_words = set()
        for chunk in retrieved_chunks:
            context_words.update(clean_text(chunk.get("text", "")))

        if not answer_words:
            return {"grounded": True, "reason": "Groundedness check passed (empty/structural response)."}

        overlap = answer_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(answer_words)

        # Require at least 8% content word overlap for groundedness
        if overlap_ratio < 0.08:
            return {
                "grounded": False,
                "reason": (
                    f"उत्तर में संदर्भ से बहुत कम शब्द मेल खाते हैं ({overlap_ratio * 100:.1f}%)। "
                    "संभावित रूप से अप्रासंगिक उत्तर। (Very low word overlap with context. Potential hallucination.)"
                ),
            }

        return {"grounded": True, "reason": "Groundedness check passed."}
