import argparse
import os
import textwrap
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Simple character-based chunking.
    Good enough for a portfolio demo; easy to understand.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap for continuity
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Use cosine similarity by normalizing vectors then inner product.
    """
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(question: str, embedder: SentenceTransformer, index: faiss.IndexFlatIP,
    chunks: List[str], top_k: int = 3, ) -> List[Tuple[int, float, str]]:


    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score), chunks[int(idx)]))
    return results


def generate_answer(question: str, context_chunks: List[str]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

    context = "\n\n---\n\n".join(context_chunks)
    prompt = textwrap.dedent(f"""
    You are a medical assistant for a synthetic demo. Use ONLY the provided context.
    If the context does not contain the answer, say: "I don't have enough information in the provided note."

    Context:
    {context}

    Question: {question}

    Answer (concise, grounded in the note):
    """).strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Clinical Notes RAG (Hugging Face + FAISS) Demo")
    parser.add_argument("--data_path", type=str, default="data/sample_clinical_note.txt")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Could not find data file: {args.data_path}")

    note_text = read_text(args.data_path)
    chunks = chunk_text(note_text, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise ValueError("No text chunks created. Check your input note file.")

    print(f"[INFO] Loaded note and created {len(chunks)} chunks.")

    print("[INFO] Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("[INFO] Embedding chunks...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")

    print("[INFO] Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("[INFO] Retrieving relevant chunks...")
    retrieved = retrieve(args.question, embedder, index, chunks, top_k=args.top_k)

    print("\n=== Retrieved Context ===")
    context_chunks = []
    for rank, (idx, score, chunk) in enumerate(retrieved, start=1):
        print(f"\n[#{rank}] chunk_id={idx} score={score:.4f}")
        # print(chunk[:800] + ("..." if len(chunk) > 800 else ""))
        context_chunks.append(chunk)

    print("\n=== Generated Answer ===")
    answer = generate_answer(args.question, context_chunks)
    print(answer)


if __name__ == "__main__":
    main()
