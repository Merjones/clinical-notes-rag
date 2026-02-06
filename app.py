import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rag import RAGEngine

app = FastAPI(title="Clinical RAG API", version="0.1")

# ---- Startup config ----
# You can set DATA_PATH to either:
# - a directory (e.g. "data") -> indexes all .txt files
# - a single file (e.g. "data/sample_clinical_note.txt")
DATA_PATH = os.getenv("DATA_PATH", "data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
OVERLAP = int(os.getenv("OVERLAP", "100"))

engine: Optional[RAGEngine] = None

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class Source(BaseModel):
    doc_id: str
    chunk_id: int
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.on_event("startup")
def startup_event():
    global engine

    if os.path.isdir(DATA_PATH):
        paths = RAGEngine.list_text_files(DATA_PATH)
        if not paths:
            raise ValueError(f"No .txt files found in directory: {DATA_PATH}")
    else:
        paths = [DATA_PATH]

    engine = RAGEngine(
        data_paths=paths,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    # Helpful server-side log
    print(
        f"[STARTUP] Indexed {len(paths)} document(s) into {engine.index.ntotal} chunk vectors "
        f"(chunk_size={CHUNK_SIZE}, overlap={OVERLAP})."
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if engine is None:
        # Should never happen if startup succeeded
        return {"answer": "Engine not initialized.", "sources": []}

    result = engine.ask(req.question, top_k=req.top_k)

    #Ensure sources match Source model exactly
    sources = [
        {"doc_id": s["doc_id"], "chunk_id": s["chunk_id"], "score": float(s["score"])}
         for s in result.get("sources", [])
    ]
    return result
