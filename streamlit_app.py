"""Streamlit Web Application for the Voice-Enabled RAG model."""

import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import VoiceRAGPipeline

# Load environment variables from .env
load_dotenv()

# App Page Configurations
st.set_page_config(
    page_title="Indic Voice Console",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Check backend keys
groq_key = os.getenv("GROQ_API_KEY", "").strip()
sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
has_backend_keys = bool(groq_key and sarvam_key)

# Inject Custom Modern CSS Styles
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Custom Glass Cards */
    .app-card {
        background: rgba(17, 25, 40, 0.70);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    
    /* Header Container */
    .header-container {
        text-align: center;
        margin-bottom: 30px;
        padding: 30px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(to right, #a5b4fc, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    /* Voice Mic Circle */
    .voice-console {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px 10px;
        margin: 20px 0;
    }
    .mic-icon-container {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #3b82f6);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.5);
        margin-bottom: 15px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .mic-icon-container:hover {
        transform: scale(1.08);
        box-shadow: 0 0 35px rgba(99, 102, 241, 0.7);
    }
    .pulse-aura {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0px rgba(99, 102, 241, 0.5); }
        100% { box-shadow: 0 0 0 25px rgba(99, 102, 241, 0); }
    }
    
    /* Status Badge styling */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        gap: 6px;
    }
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .status-demo {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    /* Grid clickable cards */
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    .sample-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .sample-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    /* Timeline / Flow Step */
    .timeline-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 15px;
    }
    .timeline-step {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        padding: 12px 18px;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
    }
    .timeline-step.passed { border-left-color: #10b981; }
    .timeline-step.failed { border-left-color: #ef4444; }
    .timeline-step.skipped { border-left-color: rgba(255, 255, 255, 0.15); opacity: 0.5; }
    
    /* Latency Speedometer */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── App Header ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="header-container">
        <h1 class="header-title">🎙️ Indic Voice Console</h1>
        <p style="margin-top: 8px; color: #9ca3af; font-size: 1.1rem; font-weight: 400;">
            Multilingual Voice-Enabled RAG model for MSMARCO-XI
        </p>
        <div style="margin-top: 15px;">
            {f'<span class="status-badge status-live">🟢 Live API Mode (Groq + Sarvam Connected)</span>' if has_backend_keys else '<span class="status-badge status-demo">🟡 Demo Mode (Simulated APIs active)</span>'}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Setup configuration overrides in sidebar if users want to tweak thresholds
st.sidebar.title("⚙️ Parameters")
strategy = st.sidebar.selectbox(
    "Chunk Strategy",
    ["metadata_aware", "semantic", "fixed_overlap", "fixed_size"],
    index=0,
)
off_topic_threshold = st.sidebar.slider(
    "Off-Topic Cutoff",
    min_value=0.20,
    max_value=0.80,
    value=0.42,
    step=0.01,
)

# Handle Manual Override Mode
force_mock = st.sidebar.checkbox(
    "Force Mock/Demo Mode",
    value=not has_backend_keys,
    disabled=not has_backend_keys,
)
is_running_mock = force_mock or not has_backend_keys

# ── Main Console Layout ──────────────────────────────────────────────────────
col_left, col_right = st.columns([7, 5])

# Preset Question cards helper
presets = {
    "q1": {"title": "🏢 निगम की परिभाषा", "text": "कॉर्पोरेशन क्या है?"},
    "q2": {"title": "🥔 पोटेशियम खाद्य पदार्थ", "text": "पोटेशियम में कम खाद्य पदार्थों का चार्ट।"},
    "q3": {"title": "📖 रेचल कार्सन दायित्व", "text": "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा"},
}

with col_left:
    st.markdown(
        """
        <div class="app-card">
            <div class="voice-console">
                <div class="mic-icon-container pulse-aura">
                    <span style="font-size: 40px; color: white;">🎙️</span>
                </div>
                <h3 style="margin-top: 10px; margin-bottom: 5px; font-weight: 600;">Voice Input Console</h3>
                <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 20px;">Upload an audio file or type a query below to prompt the RAG pipeline.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centralized input boxes
    with st.container():
        uploaded_audio = st.file_uploader(
            "Upload Spoken Query (WAV format recommended)",
            type=["wav", "mp3"],
            label_visibility="collapsed",
        )

        query_text = st.text_input(
            "Type a query directly:",
            placeholder="अपनी आवाज़ अपलोड करें या यहाँ प्रश्न टाइप करें (जैसे: कॉर्पोरेशन क्या है?)...",
            label_visibility="collapsed",
        )

        # Quick action grids
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem; font-weight: 500; margin-bottom: 5px;'>Quick Sample Prompts:</p>",
            unsafe_allow_html=True,
        )
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button(presets["q1"]["title"], use_container_width=True):
            query_text = presets["q1"]["text"]
        if col_p2.button(presets["q2"]["title"], use_container_width=True):
            query_text = presets["q2"]["text"]
        if col_p3.button(presets["q3"]["title"], use_container_width=True):
            query_text = presets["q3"]["text"]

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Execute Pipeline", use_container_width=True)

with col_right:
    st.markdown(
        """
        <div class="app-card" style="height: 100%;">
            <h4 style="margin: 0 0 15px 0; font-weight: 600;">📋 System Configuration</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                    <td style="padding: 10px 0; color: #9ca3af;">Embedding Model</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600; color: #a5b4fc;">paraphrase-multilingual-MiniLM</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                    <td style="padding: 10px 0; color: #9ca3af;">Vector Index DB</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600; color: #a5b4fc;">FAISS IP (Cosine)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                    <td style="padding: 10px 0; color: #9ca3af;">Chunking Strategy</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600; color: #38bdf8;">"""
        + strategy
        + """</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                    <td style="padding: 10px 0; color: #9ca3af;">Off-Topic Guard Cutoff</td>
                    <td style="padding: 10px 0; text-align: right; font-weight: 600; color: #38bdf8;">"""
        + str(off_topic_threshold)
        + """</td>
                </tr>
            </table>
            <div style="margin-top: 25px; padding: 12px; background: rgba(99, 102, 241, 0.08); border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.2); font-size: 0.85rem; color: #a5b4fc; text-align: center;">
                💡 Voice queries are transcribed locally/live, then matched with cosine similarity scores in FAISS.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Pipeline Run Execution ───────────────────────────────────────────────────
if run_btn:
    if not uploaded_audio and not query_text:
        st.error("Please enter a text query or upload an audio file.")
    else:
        # Load pipeline orchestrator
        pipeline = VoiceRAGPipeline(
            strategy=strategy,
            off_topic_threshold=off_topic_threshold,
        )

        audio_path = None
        if uploaded_audio:
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            audio_path = temp_dir / uploaded_audio.name
            audio_path.write_bytes(uploaded_audio.read())

        # Execute
        with st.spinner("Processing through Voice-RAG channels..."):
            res = pipeline.run_pipeline(
                audio_path=audio_path,
                query_text=None if audio_path else query_text,
                mock_stt=is_running_mock,
                mock_gen=is_running_mock,
                mock_stt_text=query_text if not audio_path else "निगम क्या है?",
            )

        status = res.get("status", "success")
        final_query = res.get("query", "")
        final_answer = res.get("answer", "")
        latencies = res.get("latencies", {})
        guards = res.get("guardrails", {})

        # Layout Response Box & Metrics
        col_res, col_metrics = st.columns([7, 5])

        with col_res:
            st.markdown(
                f"""
                <div class="app-card">
                    <h4 style="margin: 0 0 10px 0; font-weight: 600; color: #a5b4fc;">🎙️ Processed Query</h4>
                    <p style="font-size: 1.15rem; font-weight: 500; color: #f3f4f6; margin-bottom: 20px;">"{final_query}"</p>
                    
                    <h4 style="margin: 0 0 10px 0; font-weight: 600; color: #a5b4fc;">📄 System Output</h4>
                    <div style="background: rgba(255, 255, 255, 0.02); padding: 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); line-height: 1.6; font-size: 1.05rem;">
                        {final_answer}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show citations if present and not blocked
            if res.get("citations") and "blocked" not in status:
                st.markdown("**Citations & Referenced Sources:**")
                for cit in res["citations"]:
                    st.caption(f"📌 Reference ID: `{cit}`")

            # Show LLM reasoning in expander
            if res.get("reasoning"):
                with st.expander("Show Reasoning Trace"):
                    st.write(res["reasoning"])

        with col_metrics:
            # Latency breakdown box
            st.markdown(
                f"""
                <div class="app-card">
                    <h4 style="margin: 0 0 15px 0; font-weight: 600; color: #f3f4f6;">⏱️ Latency Dashboard</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.04);">
                            <div class="metric-label">STT Audio Prep</div>
                            <div class="metric-value">{latencies.get('stt', 0.0):.1f} ms</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.04);">
                            <div class="metric-label">FAISS Vector search</div>
                            <div class="metric-value">{latencies.get('retrieval', 0.0):.1f} ms</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.04);">
                            <div class="metric-label">LLM Generation</div>
                            <div class="metric-value">{latencies.get('generation', 0.0):.1f} ms</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.04);">
                            <div class="metric-label">Total Pipeline</div>
                            <div class="metric-value" style="color: #6366f1;">{latencies.get('total', 0.0):.1f} ms</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Guardrail Status visualizer
            st.markdown(
                """
                <div class="app-card">
                    <h4 style="margin: 0 0 15px 0; font-weight: 600; color: #f3f4f6;">🧱 Guardrail Verification Pipeline</h4>
                    <div class="timeline-container">
                """,
                unsafe_allow_html=True,
            )

            # Step 1: Safety Check
            safety = guards.get("safety", {})
            if safety.get("safe", True):
                st.markdown(
                    f"""
                    <div class="timeline-step passed">
                        <span>🛡️ <b>Query Safety:</b> Passed</span>
                        <span style="font-size: 0.85rem; color: #10b981;">{latencies.get('safety_guard', 0.0):.1f}ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="timeline-step failed">
                        <span>🛡️ <b>Query Safety:</b> Blocked</span>
                        <span style="font-size: 0.85rem; color: #ef4444;">{latencies.get('safety_guard', 0.0):.1f}ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Step 2: Off-topic check
            off_topic = guards.get("off_topic", {})
            if "safety" in guards and not safety.get("safe", True):
                st.markdown(
                    """
                    <div class="timeline-step skipped">
                        <span>🔍 <b>Off-topic Cutoff:</b> Skipped</span>
                        <span style="font-size: 0.85rem; color: #9ca3af;">—</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif off_topic.get("on_topic", True):
                st.markdown(
                    f"""
                    <div class="timeline-step passed">
                        <span>🔍 <b>Off-topic Cutoff:</b> Passed</span>
                        <span style="font-size: 0.85rem; color: #10b981;">{latencies.get('off_topic_guard', 0.0):.1f}ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="timeline-step failed">
                        <span>🔍 <b>Off-topic Cutoff:</b> Blocked</span>
                        <span style="font-size: 0.85rem; color: #ef4444;">{latencies.get('off_topic_guard', 0.0):.1f}ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Step 3: Groundedness check
            groundedness = guards.get("groundedness", {})
            if "off_topic" in guards and not off_topic.get("on_topic", True):
                st.markdown(
                    """
                    <div class="timeline-step skipped">
                        <span>⚖️ <b>Groundedness Verification:</b> Skipped</span>
                        <span style="font-size: 0.85rem; color: #9ca3af;">—</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif not groundedness:
                st.markdown(
                    """
                    <div class="timeline-step skipped">
                        <span>⚖️ <b>Groundedness Verification:</b> Skipped</span>
                        <span style="font-size: 0.85rem; color: #9ca3af;">—</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif groundedness.get("grounded", True):
                st.markdown(
                    f"""
                    <div class="timeline-step passed">
                        <span>⚖️ <b>Groundedness Verification:</b> Grounded</span>
                        <span style="font-size: 0.85rem; color: #10b981;">{latencies.get('groundedness_guard', 0.0):.1f}ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="timeline-step failed">
                        <span>⚖️ <b>Groundedness Verification:</b> Ungrounded</span>
                        <span style="font-size: 0.85rem; color: #ef4444;">{latencies.get('groundedness_guard', 0.0):.1f}ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Context chunks display
        st.subheader("📚 Retrieved Context Passages")
        for i, chunk in enumerate(res.get("chunks", [])):
            with st.expander(
                f"Chunk [{i}] (Similarity: {chunk.get('similarity_score', 0.0):.4f}) — Reference ID: {chunk.get('passage_id')}"
            ):
                st.markdown(chunk.get("text", ""))
                st.caption(
                    f"Selected Source Flag: `{chunk.get('is_selected')}` | Strategy: `{chunk.get('strategy')}`"
                )
