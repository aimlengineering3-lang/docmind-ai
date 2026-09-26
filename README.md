# DocMind AI

Multimodal document intelligence and RAG platform. Upload PDFs or DOCX files, ask
questions grounded in their content with page-level citations, or run structured
extraction on invoices, resumes, and contracts.

**Live demo:** https://docmind-ai-px7eiorulslps6filyqqj4.streamlit.app

## What it does

- Parses PDF and DOCX documents, preserving page/section metadata
- Chunks text and embeds it locally with BGE-small
- Retrieves relevant chunks using hybrid search — semantic (Pinecone) + keyword
  (SQLite FTS5) combined via Reciprocal Rank Fusion
- Generates grounded answers with Gemini, citing the exact document and page
- Abstains instead of guessing when the documents don't contain the answer
- Extracts structured, validated data (invoice / resume / contract) as JSON
- Exposes the same logic through both a Streamlit UI and a FastAPI backend

## Architecture

Document → parsing (PyMuPDF / python-docx) → chunking → BGE-small embeddings
→ Pinecone (semantic index) + SQLite FTS5 (keyword index)
→ hybrid retrieval (RRF)
→ Gemini → grounded answer + citations, or structured JSON extraction

Streamlit UI ──┐
├─→ shared service layer (Index, qa, extraction)
FastAPI API ───┘


The UI and API don't duplicate logic — both call the same `Index`, `answer_question`,
and `extract_structured` functions. Either one can be swapped or extended without
touching the other.

## Tech stack and why

| Component | Choice | Why |
|---|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 | Runs locally on CPU, no API cost, good quality for its size |
| Vector store | Pinecone (serverless, free tier) | Managed, production-realistic, free tier covers this project's scale many times over |
| Keyword search | SQLite FTS5 | Free, local, no reason to pay for what a built-in index already does well |
| LLM | Gemini (`gemini-flash-lite-latest`) | Free tier with a usable daily quota; the task is grounded generation over retrieved text, not open-ended reasoning, so a Flash-tier model is sufficient |
| Backend | FastAPI | Decouples the service layer from the UI so any future client (mobile, another app) can reuse it |
| Frontend | Streamlit | Fast to build, good enough for a live interview demo |
| Validation | Pydantic | Structured extraction output is schema-checked, not raw LLM JSON |

No paid APIs, no GPU, no infrastructure beyond what's listed above. Runs on an
8 GB, CPU-only laptop.

## Safety / grounding design

Two layers stop the system from making things up:
1. A retrieval-score threshold — if nothing relevant enough was retrieved, the
   LLM is never even called.
2. A system prompt instructing the model to answer only from the given context
   and say so plainly when it can't.

Both are exercised in the evaluation set below.

## Evaluation

A 10-question set spanning three documents (an invoice, a resume, and a
15-page research paper), checked for retrieval accuracy, answer correctness,
and correct abstention on an out-of-scope question:

Retrieval hit rate: 9/9
Answer keyword rate: 9/9
Abstention accuracy: 10/10


Run it yourself: `python -m docmind.eval`

## Running locally

```bash
git clone https://github.com/aimlengineering3-lang/docmind-ai.git
cd docmind-ai
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:

GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=docmind


```bash
streamlit run app.py                          # UI
python -m uvicorn docmind.api:app --reload     # API (optional, separate process)
```

## Testing

```bash
python -m pytest -v
```
32 tests, all offline (fake embedder/vector-store/LLM — no network calls, no cost).

## Known limitations

- **No OCR yet**: scanned/image-only PDFs are detected and reported clearly to
  the user rather than silently failing. PaddleOCR integration is a planned
  next step, deferred to keep the local footprint small on an 8 GB machine.
- **Abstention threshold** is a coarse pre-filter, not perfectly calibrated —
  the system-prompt-level refusal is the real safety net and was correct in
  100% of the evaluation cases.