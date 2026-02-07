# Clinical RAG API
Retrieval-Augmented Generation (RAG) service for querying clinical notes with grounded, citation-like references, served via FastAPI.

## Overview
This repo implements a lightweight Clinical RAG pipeline: it embeds clinical text, retrieves the most relevant chunks using vector similarity search, and generates an answer conditioned on the retrieved context.

It is designed as a clean, backend-first portfolio project demonstrating how to productionize an LLM workflow behind an API (no front-end required).

## Problem
Standalone LLMs can produce confident but incorrect outputs (hallucinations) and typically cannot cite where facts came from. In clinical and scientific settings, answers must be:
- grounded in source text
- traceable to evidence
- reproducible and testable

## Solution
This project uses a RAG approach:
1) split a clinical note into chunks  
2) embed each chunk into a vector space  
3) retrieve top-k relevant chunks for a user question  
4) pass retrieved context into a generation model  
5) return an answer + the retrieved text that supported it

## Architecture 
User Query
    ↓
FastAPI Endpoint (/ask)
    ↓
Embedding Model
    ↓
FAISS Vector Search
    ↓
Top-k Clinical Chunks
    ↓
LLM (context-aware prompt)
    ↓
Answer + Sources

## Tech Stack 
- Python 
- FastAPI + Uvicorn
- FAISS
- SentenceTransformers embeddings
- Hugging Face Transformers (FLAN-T5)

## Models
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Generation model:** `google/flan-t5-base`

## Features
- Load and chunk a clinical note from `data/`
- Embed chunks and retrieve relevant context per query
- Generate answers conditioned on retrieved text
- FastAPI backend with interactive Swagger docs (`/docs`)
- 
## Project Structure 
clinical-rag/
│
├── app/
│   ├── main.py          # Entrypoint
│   ├── app.py           # FastAPI server and routes 
│   ├── rag.py           # RAGEngine (chunking, retrieval, generation)
│
├── data/
│   └── sample_clinical_note.txt
├── requirements.txt
└── README.md

## Usage 
 ### Option 1 - CLI (Batch/local querying) 
 - python main.py --data_path data --question "What is the assessment and plan?"
 ### Option 2 - API Server 
 - run as a backend service
 - start server: uvicorn app:app --reload
 - interactive Swagger UI available (/docs)

   

