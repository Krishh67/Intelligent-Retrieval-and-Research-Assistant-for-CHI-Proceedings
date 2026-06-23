# 🔬 Intelligent Retrieval and Research Assistant for CHI Proceedings

> A production-ready Retrieval-Augmented Generation (RAG) system for searching and synthesizing research from **2,635 ACM CHI papers** (2021, 2023, 2024) — powered by FAISS dense retrieval, BAAI/bge-large-en-v1.5 embeddings, and Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?logo=google)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0064E0)](https://faiss.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Pipeline Stages](#-pipeline-stages)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Pre-built Embeddings](#-pre-built-embeddings)
- [Running the App](#-running-the-app)
- [Building Embeddings from Scratch](#-building-embeddings-from-scratch)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview

This project provides an **intelligent research assistant** for the ACM CHI (Conference on Human Factors in Computing Systems) corpus. Users can ask natural language research questions and receive grounded, cited answers synthesized from real CHI papers.

**Key capabilities:**
- 🔍 **Multi-query dense retrieval** — expands every user question into 1–6 semantically diverse sub-queries
- 📅 **Year-aware filtering** — automatically detects year intent (e.g. *"CHI 2024 papers on…"*) and filters results
- 🤖 **LLM-grounded answers** — Gemini synthesizes answers citing specific papers with publication years
- ⚡ **Quota-resilient** — automatic fallback across multiple API keys and model variants
- 🎨 **Polished Streamlit UI** — editorial dark-mode interface with interactive source cards and year distribution chart

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Planner     │  ← 1 Gemini call (gemini-2.5-flash)
│  query_planner.py   │    Detects: topic · year · filter flag
│                     │    Generates: 1–6 retrieval sub-queries
└──────────┬──────────┘
           │  QueryPlan (JSON)
           ▼
┌─────────────────────┐
│  Year-Boosted       │  ← Appends "CHI {year}" to queries
│  Queries            │    when a year is detected (pre-retrieval)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Multi-Query FAISS  │  ← N parallel BAAI/bge-large-en-v1.5
│  Retrieval          │    cosine similarity searches
│  retrieval_v2.py    │    (IndexFlatIP · 75,817 vectors)
└──────────┬──────────┘
           │  Raw results (all queries merged)
           ▼
┌─────────────────────┐
│  Cross-Query Merge  │  ← Keep highest score per paper_id
│  & Deduplication    │    Sort descending by score
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Metadata Year      │  ← Post-retrieval year filter
│  Filter             │    Falls back to unfiltered if < 3 remain
└──────────┬──────────┘
           │  Top-8 papers
           ▼
┌─────────────────────┐
│  Context Builder    │  ← Formats paper blocks (~14K tokens)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Gemini Answer      │  ← 1 Gemini call — cited, grounded answer
│  Generation         │
└─────────────────────┘
```

**Total Gemini API calls per user question: 2** (planner + answer generation).

---

## 📚 Dataset

| Property | Value |
|---|---|
| Conference | ACM CHI (Human Factors in Computing Systems) |
| Years covered | 2021 · 2023 · 2024 |
| Total papers | **2,635** |
| Total sections | **21,802** |
| Total chunks | **75,817** |
| Chunk size | 700 words with 100-word overlap |
| Embedding model | `BAAI/bge-large-en-v1.5` (768-dim) |
| Vector index | FAISS `IndexFlatIP` (cosine similarity) |
| Index size | ~296 MB |

Papers were extracted from official ACM CHI PDF proceedings using PyMuPDF, parsed into section-aware chunks, and embedded with a state-of-the-art semantic search model.

---

## 🔧 Pipeline Stages

### Stage 1 — PDF Extraction (`paper_n_section_extraction/`)
Extracts papers and sections from raw CHI PDF proceedings:
- `scripts/00_run_pipeline.py` — end-to-end PDF → JSON pipeline (PyMuPDF-based)
- `final/04_merge_corpus.py` — merges per-year JSON files into a unified corpus with globally unique paper IDs and year metadata

### Stage 2 — Chunking (`05_chunk_sections.py`)
Section-aware word-based chunking (700 words / 100-word overlap). Chunks never cross section boundaries to preserve semantic structure.

### Stage 3 — Embedding Generation
Two options:
- `06_generate_embeddings.py` — local CPU/GPU generation
- `06_COLAB_generate_embeddings.py` — Google Colab optimised (GPU, checkpointing every 10K chunks)

Produces three artifacts saved to `embedding/`:
| File | Size | Description |
|---|---|---|
| `faiss.index` | ~296 MB | FAISS IndexFlatIP (L2-normalized) |
| `embeddings.npy` | ~296 MB | Raw embedding array (75817 × 768) |
| `metadata.pkl` | ~5 MB | Per-chunk metadata (paper_id, year, title, section) |
| `chunks.jsonl` | ~303 MB | Full chunk text corpus |

### Stage 4 — Retrieval (`retrieval_v2.py`)
`CHIRetriever` class: embed query → FAISS search → paper-level deduplication → `RetrievalResult` dataclass list.

### Stage 5 — RAG Pipeline (`9_rag_pipeline.py`)
`CHIResearchAssistant.ask(question)` — the full end-to-end pipeline described in the architecture diagram above.

### Stage 6 — Streamlit Frontend (`app.py`)
Dark-mode research assistant UI with:
- Query form + 5 example chips
- Sources panel with relevance score bars and year distribution chart
- Retrieval process inspector (collapsible)
- Multi-key / multi-model quota management in settings sidebar

---

## 📁 Project Structure

```
.
├── app.py                              # Streamlit frontend (main entry point)
├── 9_rag_pipeline.py                   # Core RAG pipeline (CHIResearchAssistant)
├── retrieval_v2.py                     # FAISS retrieval layer (CHIRetriever)
├── query_planner.py                    # Gemini query planner (analyze_and_expand_query)
│
├── 05_chunk_sections.py                # Section-aware chunking
├── 06_generate_embeddings.py           # Local embedding generation
├── 06_COLAB_generate_embeddings.py     # Google Colab embedding generation
├── 07_testing_retrieval_bulk.py        # Bulk retrieval benchmark (40 test queries)
│
├── paper_n_section_extraction/
│   ├── scripts/
│   │   └── 00_run_pipeline.py          # PDF → JSON extraction pipeline
│   └── final/
│       └── 04_merge_corpus.py          # Multi-year corpus merger
│
├── embedding/                          # ← NOT in git (see Pre-built Embeddings)
│   ├── faiss.index
│   ├── embeddings.npy
│   ├── metadata.pkl
│   └── chunks.jsonl
│
├── requirements.txt
├── .env                                # ← NOT in git (add your API keys here)
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Google Gemini API key(s) — [get one free](https://aistudio.google.com/apikey)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Krishh67/Intelligent-Retrieval-and-Research-Assistant-for-CHI-Proceedings.git
cd Intelligent-Retrieval-and-Research-Assistant-for-CHI-Proceedings

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** For faster embedding generation, replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` if you have a CUDA-capable GPU.

---

## 🔑 Configuration

Create a `.env` file in the project root:

```env
# Primary API key (required)
GOOGLE_API_KEY1=your_gemini_api_key_here

# Additional keys for automatic quota fallback (optional)
GOOGLE_API_KEY2=your_second_key_here
GOOGLE_API_KEY3=your_third_key_here
```

The app supports **up to 9 API keys** (`GOOGLE_API_KEY1` … `GOOGLE_API_KEY9`) and will automatically rotate through them and across model variants when quota limits are hit.

---

## 📦 Pre-built Embeddings

> The `embedding/` folder (~900 MB total) is not included in this repository due to GitHub file size limits.

**Download the pre-built embedding artifacts here:**

> 🔗 **[Embedding Folder — Google Drive](https://drive.google.com/drive/folders/1e3YVjEJCSol7AFkVsg-Eo62ieLa5qMTW?usp=sharing)**

After downloading, place the files as follows:

```
embedding/
├── faiss.index      (~296 MB)
├── embeddings.npy   (~296 MB)
├── metadata.pkl     (~5 MB)
└── chunks.jsonl     (~303 MB)
```

These files are required to run the app. Without them, the retriever will fail to initialise.

---

## 🚀 Running the App

Once the `embedding/` folder is in place and `.env` is configured:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 🏗 Building Embeddings from Scratch

If you have access to the CHI PDF proceedings and want to build the index yourself:

### Step 1 — Extract papers from PDFs
```bash
cd paper_n_section_extraction/scripts
python 00_run_pipeline.py
```

### Step 2 — Merge yearly JSON files
```bash
cd paper_n_section_extraction/final
python 04_merge_corpus.py
```

### Step 3 — Chunk the corpus
```bash
python 05_chunk_sections.py
```

### Step 4 — Generate embeddings

**Option A — Local (CPU/GPU):**
```bash
python 06_generate_embeddings.py
```

**Option B — Google Colab (recommended for GPU speed):**  
Upload `06_COLAB_generate_embeddings.py` to Colab and follow the cell-by-cell instructions in the file header. Saves checkpoints every 10,000 chunks to Google Drive.

### Step 5 — Test retrieval
```bash
python 07_testing_retrieval_bulk.py
```
Runs 40 benchmark queries across 10 topic categories and writes a full report to `embedding/retrieval_report_2.txt`.

---

## 📖 API Reference

### `CHIResearchAssistant.ask(question: str) → RAGResponse`

```python
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("rag_pipeline", "9_rag_pipeline.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assistant = mod.CHIResearchAssistant(mod.RAGConfig())
response  = assistant.ask("What accessibility solutions have been proposed for users with disabilities?")

print(response.answer)
for paper in response.papers:
    print(f"  [{paper.year}] {paper.title}  (score: {paper.score:.3f})")
```

### `RAGConfig` parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `gemini-2.5-flash` | Gemini model for answer generation |
| `planner_model` | `gemini-2.5-flash` | Gemini model for query planning |
| `temperature` | `0.6` | Generation temperature |
| `max_output_tokens` | `2048` | Max tokens in generated answer |
| `faiss_path` | `embedding/faiss.index` | Path to FAISS index |
| `per_query_k` | `5` | Papers returned per sub-query |
| `per_query_retrieve_k` | `20` | Chunks retrieved per sub-query before dedup |
| `final_top_k` | `8` | Max papers sent to context builder |
| `fallback_min_papers` | `3` | Min papers required after year filter before fallback |

---

## 🖼 Screenshots

> *(Add screenshots of the Streamlit UI here)*

---

## 🙏 Acknowledgements

- **ACM CHI** — proceedings corpus
- **BAAI** — [`bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) embedding model
- **Meta AI** — [FAISS](https://faiss.ai/) vector search library
- **Google DeepMind** — [Gemini](https://ai.google.dev/) language model
- **NTNU** — project supervision and research context

---

<p align="center">
  Built with ❤️ for HCI research · ACM CHI 2021 · 2023 · 2024
</p>
