import argparse
import os

from rag import RAGEngine


def main():
    parser = argparse.ArgumentParser(description="Clinical Notes RAG (HF + FAISS) Demo")
    parser.add_argument("--data_path", type=str, default="data")  # now can be file or directory
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    # Support: pass a directory (index all .txt files) OR a single file
    if os.path.isdir(args.data_path):
        paths = RAGEngine.list_text_files(args.data_path)
        if not paths:
            raise ValueError(f"No .txt files found in directory: {args.data_path}")
    else:
        paths = [args.data_path]

    engine = RAGEngine(
        data_paths=paths,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"[INFO] Indexed {len(paths)} document(s) into {engine.index.ntotal} chunk vectors.")

    result = engine.ask(args.question, top_k=args.top_k)

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Sources ===")
    for s in result["sources"]:
        print(f'- doc="{s["doc_id"]}" chunk_id={s["chunk_id"]} score={s["score"]:.4f}')


if __name__ == "__main__":
    main()
