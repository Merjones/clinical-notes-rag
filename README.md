# clinical-notes-rag

A minimal Retrieval-Augmented Generation (RAG) demo for **synthetic** clinical notes using:
- Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- FAISS vector search
- Hugging Face text generation (`google/flan-t5-base`)

This repo focuses on practical engineering: chunking, retrieval, grounded generation, and clear outputs.

## Important
All example notes are **synthetic**. Do not add PHI.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt