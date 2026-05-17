"""
rag/build_index.py
──────────────────
Construit l'index FAISS à partir du corpus de principes éthiques
(data/ethical_corpus.json). Persiste l'index + métadonnées pour
être utilisé par rag/inference.py.

Embedder : sentence-transformers (BAAI/bge-small-en — léger, ~30 MB).

Usage :
  python rag/build_index.py --corpus data/ethical_corpus.json --out_dir rag/corpus
"""

import argparse
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    docs = data.get("documents", [])
    print(f"Loaded {len(docs)} documents from {path}")
    return docs


def build_index(docs: list[dict], embedder_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading embedder: {embedder_name}")
    embedder = SentenceTransformer(embedder_name)

    # On encode le texte (le titre peut aussi être préfixé pour booster la recherche)
    texts = [f"{d.get('title', '')}: {d['text']}" for d in docs]
    print("Encoding…")
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    # Index inner product sur embeddings normalisés = cosine similarity
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print(f"FAISS index built with {index.ntotal} vectors, dim={dim}")

    # Save
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"embedder": embedder_name, "n_docs": len(docs), "dim": dim}, f, indent=2)
    print(f"Saved index + docs -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/ethical_corpus.json")
    parser.add_argument("--out_dir", type=str, default="rag/corpus")
    parser.add_argument("--embedder", type=str, default="BAAI/bge-small-en")
    args = parser.parse_args()
    docs = load_corpus(args.corpus)
    build_index(docs, args.embedder, args.out_dir)
