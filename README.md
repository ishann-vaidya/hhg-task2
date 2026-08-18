# Voice-Enabled RAG — HH Goa 2026 · Task 2

Voice input → STT (Sarvam) → Retrieval (FAISS + sentence-transformers) → Generation (Groq) with guardrails and latency analytics.

**Dataset:** [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (Indic-language MS MARCO passages)

## Phase 0 — Setup & Data (current)

### Prerequisites

- Python 3.10+ (tested on 3.13)
- ~2 GB free disk for a Hindi validation subset; full dataset is ~55 GB

### Quick start

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# Windows (Git Bash / CMD)
source .venv/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 0 exploration (downloads dataset on first run)
python scripts/phase0_explore_dataset.py
```

## Phase 1 — Chunking Strategy Comparison

Run side-by-side comparison of 4 chunking strategies:
```bash
# Compare on 3 sample passages and save outputs to data/processed/chunks/
python scripts/phase1_compare_chunking.py --passages 3 --save

# Run unit tests
pytest tests/
```

### Configuration

Edit `config/settings.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `DEFAULT_LANGUAGE` | `"hi"` | Hindi — good demo language with broad tooling |
| `DEFAULT_SPLIT` | `"validation"` | Smaller than train |
| `SUBSET_SIZE` | `2000` | Examples to index for the hackathon |
| `CHUNK_SIZE_TOKENS` | `256` | Base chunk size limit |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap for continuity |

### Project layout

```
hhg-task2/
├── config/settings.py          # Central config
├── src/
│   ├── data/loader.py          # Dataset loading utilities
│   └── chunking/               # Base models, 4 chunking strategies & pipeline
│       ├── base.py
│       ├── strategies.py
│       └── pipeline.py
├── scripts/                    # Phase comparison & exploration scripts
│   ├── phase0_explore_dataset.py
│   └── phase1_compare_chunking.py
├── tests/                      # Automated test suite
│   └── test_chunking.py
├── data/                       # Generated data & indexes (gitignored)
└── requirements.txt
```

## Build phases

- [x] **Phase 0:** Repo + env + dataset loaded
- [x] **Phase 1:** Chunking strategies (Fixed-size, Fixed-overlap, Semantic, Metadata-aware)
- [ ] Phase 2: Embeddings + FAISS index
- [ ] Phase 3: Retrieval + latency baseline
- [ ] Phase 4: STT integration
- [ ] Phase 5: Generation + harness
- [ ] Phase 6: Guardrails
- [ ] Phase 7: Latency analytics
- [ ] Phase 8: Deploy

