# Indic Voice RAG Console — HH Goa 2026 (Task 2)

A production-grade, decoupled, voice-enabled Retrieval-Augmented Generation (RAG) assistant for Indian languages. The system transcribes spoken user audio, performs semantic vector retrieval, executes localized safety and groundedness guardrails, and synthesizes responses in the target script in under 400ms.

---

## 🏗️ System Architecture

The project is structured as a fully decoupled service:
1. **Backend Microservice (FastAPI)**: Handles speech transcription (STT), FAISS vector search, Groq LLM orchestration, custom guardrails, and latency benchmarking.
2. **Frontend Console (React + Vite + Tailwind CSS)**: An interactive user dashboard displaying real-time waveform states, timeline process flows, parameter tuning, citation cards, and LLM reasoning steps, backed by an interactive canvas particle background.

---

## 🚀 Key Features

- **Decoupled Architecture**: Independent Python backend + React frontend client.
- **Multilingual Scope**: Core interface in English for accessibility, with full end-to-end voice query and response synthesis in **English, Hindi, Marathi, Telugu, and Tamil**.
- **Interactive Background Animation**: High-performance canvas animation showing floating neural connections and voice waveforms that react to user recording and processing states.
- **Robust Localized Guardrails**:
  - *Input Safety*: Audits inputs against toxic keywords.
  - *Topic Scope Cutoff*: Similarity threshold filters (adjustable on the fly) to block off-topic queries.
  - *Output Groundedness*: Word-overlap verification to detect LLM hallucinations.
  - *Dynamic Localization*: Warning messages are generated in the user's selected language.
- **Latency Instrumentation**: Active telemetry logging presenting real-time P50/P70/P99 execution metrics on the dashboard.

---

## 📁 Project Layout

```
hhg-task2/
├── main.py                     # FastAPI API entrypoint & routes
├── config/settings.py          # Central system configuration
├── src/                        # Python Core RAG codebase
│   ├── data/loader.py          # Multi-language dataset loader
│   ├── chunking/               # Text chunking strategies (Fixed, Semantic, etc.)
│   ├── retrieval/retriever.py   # FAISS Index retriever
│   ├── stt/sarvam.py           # Sarvam AI STT locale-mapped transcriber
│   ├── generation/generator.py # Groq model orchestrator (groq/compound-mini)
│   ├── guardrails/guard.py     # Localized safety & groundedness guardrails
│   └── pipeline.py             # Orchestrator combining RAG steps
├── scripts/                    # Utility scripts (index builder, explore data)
│   ├── phase0_explore_dataset.py
│   ├── phase1_compare_chunking.py
│   ├── phase2_build_index.py   # Builds FAISS indexes per language
│   └── phase7_benchmark.py     # Benchmarks latency percentiles
├── tests/                      # Python automated test suite
├── frontend/                   # React Client Codebase
│   ├── src/App.jsx             # React dashboard & landing page
│   ├── src/index.css           # Tailwind base styles
│   ├── package.json            
│   └── vite.config.js          
├── requirements.txt            # Python dependencies
└── .env                        # Local API credentials (Groq & Sarvam)
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js (v18+) & npm

### 2. Python Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # (Or .venv\Scripts\Activate.ps1 on Windows PowerShell)

# Install dependencies
pip install -r requirements.txt

# Configure your API credentials inside .env (Root folder)
# Edit .env and insert:
# GROQ_API_KEY="gsk_..."
# SARVAM_API_KEY="sk_..."
```

### 3. Build Vector Indexes (Hindi & Marathi)
Build the local vector indexes on the validation subset (e.g. `n = 100` examples for quick setup, or `2000` for presentation datasets):
```bash
python scripts/phase2_build_index.py --language en --n 100
python scripts/phase2_build_index.py --language hi --n 100
python scripts/phase2_build_index.py --language mr --n 100
```

### 4. Run Backend API Server
Start the FastAPI server on port 8000:
```bash
python -m uvicorn main:app --reload --port 8000
```
*Swagger API documentation will be available at `http://localhost:8000/docs`.*

### 5. Frontend Console Setup
In a new terminal window, compile and run the React client:
```bash
cd frontend
npm install
npm run dev -- --force
```
*Open `http://localhost:5173` to launch the Landing Page and RAG Console!*

---

## 🧪 Testing & Validation

### Backend Python Test Suite
Run the 16 unit and integration test cases verifying safety checks, indexing, and FastAPI route responses:
```bash
pytest tests/
```

### Latency Percentile Benchmarking
Run the benchmark suite to test your local hardware speeds and generate the latency dashboard report:
```bash
python scripts/phase7_benchmark.py --n 20
```

---

## 🏁 Build Phases

- [x] **Phase 0:** Setup environment and explore Hugging Face dataset.
- [x] **Phase 1:** Comparative implementation of 4 text chunking strategies.
- [x] **Phase 2:** Dynamic embeddings generation and FAISS vector indices construction.
- [x] **Phase 3:** Retrieval pipeline implementation & latency metrics tracking.
- [x] **Phase 4:** Sarvam AI STT integration with locale code mapping.
- [x] **Phase 5:** Structured Groq LLM integration and generation harness.
- [x] **Phase 6:** Localized Input Safety, Off-topic cutoff, and Groundedness guardrails.
- [x] **Phase 7:** Automated Latency Analytics benchmark suite.
- [x] **Phase 8:** Decoupled deployment wrapper (FastAPI API server + React Client).
