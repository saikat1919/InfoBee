---
title: ShikkhaBondhu
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ShikkhaBondhu - PDF Question Answering

Upload one or more PDFs and ask questions about them. Built with:

- **Ingestion:** Markitdown + Camelot (tables)
- **Retrieval:** Hybrid search — ChromaDB (dense/semantic) + BM25 (sparse/keyword),
  combined via weighted Reciprocal Rank Fusion
- **Generation:** Conversational, history-aware query rewriting + LLM answer generation with reasoning when needed
- **Interface:** Streamlit

Everything is processed in-memory per session — nothing is persisted once the
session ends.

## Setup (local development)

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your actual API keys
3. `streamlit run app.py`

## Required environment variables

See `.env.example` and replace it with .env where you'll need:
- `HF_TOKEN`: Create a READ scope HF token from your Huggingface account.
- `GROQ_API_KEY`: Create a new API token from https://console.groq.com/keys

## 🌐 Live Demo
👉 https://huggingface.co/spaces/saikat007/ShikkhaBondhu