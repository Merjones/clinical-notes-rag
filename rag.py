from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"


@dataclass
class ChunkRecord:
    """Metadata + text for one chunk."""
    doc_id: str           # e.g., filename like "note1.txt"
    chunk_id: int         # chunk index within that doc
    text: str


class RAGEngine:
    """
    Minimal RAG engine:
      - loads docs
      - chunks docs
      - embeds chunks
      - builds FAISS index
      - retrieves top-k chunks for a question
      - generates an answer using an LLM (FLAN-T5)
    """

    def __init__(self, data_paths: List[str], chunk_size: int = 500, overlap: int = 100,
        embed_model_name: str = EMBED_MODEL_NAME, gen_model_name: str = GEN_MODEL_NAME,):
        self.data_paths = data_paths
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Load models once (important for API later)
        self.embedder = SentenceTransformer(embed_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

        # Build index from the provided documents
        self.records: List[ChunkRecord] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._build_index()

    # ---------- File loading ---------- (utility method)
    @staticmethod
    def read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def list_text_files(data_dir: str) -> List[str]:
        """Convenience: load all .txt files from a directory."""
        paths = []
        for name in os.listdir(data_dir):
            if name.lower().endswith(".txt"):
                paths.append(os.path.join(data_dir, name))
        paths.sort()
        return paths

    # ---------- Chunking ---------- (utility method)
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Simple character-based chunking with overlap.
        Preserves characters exactly; may split mid-word/sentence.
        """
        text = text.strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
            if start < 0:
                start = 0
            if end == len(text):
                break
        return chunks

    # ---------- Index building ---------- (Internal to engine)
    def _build_index(self) -> None:
        """Build FAISS index over all chunks from all documents."""
        self.records = []

        # 1) Collect chunk records for each doc
        for path in self.data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Could not find data file: {path}")

            doc_text = self.read_text(path)
            doc_chunks = self.chunk_text(doc_text, chunk_size=self.chunk_size, overlap=self.overlap)

            doc_id = os.path.basename(path)
            for i, ch in enumerate(doc_chunks):
                self.records.append(ChunkRecord(doc_id=doc_id, chunk_id=i, text=ch))

        if not self.records:
            raise ValueError("No chunks found across provided documents.")

        # 2) Embed all chunk texts
        texts = [r.text for r in self.records]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")

        # 3) Normalize for cosine-sim-as-inner-product
        faiss.normalize_L2(embeddings)

        # 4) Build index and add vectors in same order as records
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index

    # ---------- Retrieval ---------- (Public API)
    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
          { doc_id, chunk_id, score, text }
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not built.")

        q_emb = self.embedder.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)  # in-place

        scores, ids = self.index.search(q_emb, top_k)

        out: List[Dict[str, Any]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            rec = self.records[int(idx)]
            out.append(
                {
                    "doc_id": rec.doc_id,
                    "chunk_id": rec.chunk_id,
                    "score": float(score),
                    "text": rec.text,
                }
            )
        return out

    # ---------- Generation ---------- (Public API)
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        prompt = textwrap.dedent(
            f"""
            You are a medical assistant for a synthetic demo. Use ONLY the provided context.
            If the context does not contain the answer, say: "I don't have enough information in the provided note."

            Context:
            {context}

            Question: {question}

            Answer (concise, grounded in the note):
            """
        ).strip()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ---------- End-to-end ---------- (Public API)
    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        End-to-end RAG:
          - retrieve
          - generate
        Returns:
          { answer, sources: [{doc_id, chunk_id, score}] }
        """
        retrieved = self.retrieve(question, top_k=top_k)
        context_chunks = [r["text"] for r in retrieved]

        # Basic guardrail (you can improve later with score thresholds)
        if not context_chunks:
            return {
                "answer": "I don't have enough information in the provided note.",
                "sources": [],
            }

        answer = self.generate_answer(question, context_chunks)
        sources = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "score": r["score"]} for r in retrieved]
        return {"answer": answer, "sources": sources}
