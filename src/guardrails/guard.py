"""Guardrails for safety, off-topic queries, and output groundedness verification."""

import re
from typing import Any


class RAGGuardrails:
    """Implements safety filters, off-topic detection, and hallucination checks with localized warnings."""

    def __init__(self, off_topic_threshold: float = 0.42, language: str = "hi"):
        self.off_topic_threshold = off_topic_threshold
        self.language = language

        # Heuristic unsafe phrase pattern for demonstration (English & Hindi)
        self.unsafe_pattern = re.compile(
            r"\b(बम|हत्या|गाली|suicide|kill|murder|bomb|abuse|fuck|terrorist|आतंकवादी)\b",
            re.IGNORECASE,
        )

        # Localized safety messages
        self.safety_messages = {
            "en": "Query contains unsafe or inappropriate language.",
            "hi": "गाली-गलौज या असुरक्षित भाषा पाई गई। (Query contains unsafe or inappropriate language.)",
            "mr": "अपशब्द किंवा असुरक्षित भाषा आढळली। (Query contains unsafe or inappropriate language.)",
            "te": "అసురక్షిత లేదా తగని భాష కనుగొనబడింది। (Query contains unsafe or inappropriate language.)",
            "ta": "பாதுகாப்பற்ற அல்லது பொருத்தமற்ற மொழி கண்டறியப்பட்டது। (Query contains unsafe or inappropriate language.)",
        }

        # Localized no context messages
        self.no_context_messages = {
            "en": "No relevant context found. Query is off-topic.",
            "hi": "कोई प्रासंगिक जानकारी नहीं मिली। (No relevant context found. Query is off-topic.)",
            "mr": "कोणतीही संबंधित माहिती आढळली नाही। (No relevant context found. Query is off-topic.)",
            "te": "సంబంధిత సమాచారం కనుగొనబడలేదు। (No relevant context found. Query is off-topic.)",
            "ta": "தொடர்புடைய சூழல் எதுவும் கிடைக்கவில்லை। (No relevant context found. Query is off-topic.)",
        }

        # Localized off-topic messages
        self.off_topic_messages = {
            "en": "This question is out of scope of our knowledge base.",
            "hi": "यह प्रश्न हमारी जानकारी के दायरे से बाहर है।",
            "mr": "हा प्रश्न आमच्या माहितीच्या कक्षेबाहेर आहे।",
            "te": "ఈ ప్రశ్న మా పరిజ్ఞాన పరిధికి వెలుపల ఉంది।",
            "ta": "இந்த கேள்வி எங்கள் அறிவுத் தளத்தின் எல்லைக்கு அப்பாற்பட்டது।",
        }

        # Localized groundedness messages
        self.groundedness_messages = {
            "en": "Answer is not grounded in the provided context.",
            "hi": "प्रदान किए गए संदर्भ में उत्तर समर्थित नहीं है। (LLM flagged response as not grounded.)",
            "mr": "उत्तर दिलेल्या संदर्भात समर्थित नाही। (LLM flagged response as not grounded.)",
            "te": "సమాధానం అందించిన సందర్భంలో మద్దతు లేదు। (LLM flagged response as not grounded.)",
            "ta": "விடை வழங்கப்பட்ட சூழலில் ஆதரிக்கப்படவில்லை। (LLM flagged response as not grounded.)",
        }

        # Localized low overlap messages
        self.low_overlap_messages = {
            "en": "Answer has very low word overlap with context. Potential hallucination.",
            "hi": "उत्तर में संदर्भ से बहुत कम शब्द मेल खाते हैं {ratio:.1f}%। (Very low word overlap with context. Potential hallucination.)",
            "mr": "उत्तरामध्ये संदर्भाशी खूप कमी शब्द जुळतात {ratio:.1f}%। (Very low word overlap with context. Potential hallucination.)",
            "te": "సమాధానం సందర్భంతో చాలా తక్కువ పదాల అతివ్యాప్తి కలిగి ఉంది {ratio:.1f}%। (Very low word overlap with context. Potential hallucination.)",
            "ta": "விடை சூழலுடன் மிகக் குறைந்த வார்த்தை ஒன்றுடன் ஒன்று பொருந்துகிறது {ratio:.1f}%। (Very low word overlap with context. Potential hallucination.)",
        }

    def verify_query(self, query: str) -> dict[str, Any]:
        """Screen query for safety violations."""
        if self.unsafe_pattern.search(query):
            msg = self.safety_messages.get(self.language, self.safety_messages["en"])
            return {"safe": False, "reason": msg}
        return {"safe": True, "reason": "Query is safe."}

    def verify_retrieval(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Detect off-topic queries by checking retrieval similarity scores against a threshold."""
        if not retrieved_chunks:
            msg = self.no_context_messages.get(self.language, self.no_context_messages["en"])
            return {"on_topic": False, "reason": msg}

        # Check similarity of the best retrieved chunk
        best_score = retrieved_chunks[0].get("similarity_score", 0.0)
        if best_score < self.off_topic_threshold:
            base_msg = self.off_topic_messages.get(self.language, self.off_topic_messages["en"])
            msg = f"{base_msg} (Similarity: {best_score:.4f} < {self.off_topic_threshold}). (Query is off-topic.)"
            return {"on_topic": False, "reason": msg}

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
            msg = self.groundedness_messages.get(self.language, self.groundedness_messages["en"])
            return {"grounded": False, "reason": msg}

        # Heuristic word overlap verification to catch completely ungrounded text
        def clean_text(text: str) -> set[str]:
            words = re.findall(r"\w+", text.lower())
            # Basic stopwords
            stops = {
                "है", "हैं", "का", "की", "के", "में", "से", "को", "और", "यह", "वह", "जो", "तो", "पर", "हो", "भी", "एक",
                "is", "the", "are", "of", "and", "in", "to", "a", "that", "it", "was", "for", "on", "with", "as", "at",
                "आहे", "आहेत", "चा", "ची", "चे", "ने", "वर", "पण", "आणि", "या", "ते", "ती", "तो"
            }
            return {w for w in words if w not in stops and len(w) > 1}

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
            template = self.low_overlap_messages.get(self.language, self.low_overlap_messages["en"])
            msg = template.format(ratio=overlap_ratio * 100)
            return {"grounded": False, "reason": msg}

        return {"grounded": True, "reason": "Groundedness check passed."}
