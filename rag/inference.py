"""
rag/inference.py
────────────────
Inférence RAG : pour chaque requête, récupère top-k principes éthiques pertinents
depuis l'index FAISS et les injecte dans le prompt avant la génération.

Expose une classe `EthicsRetriever` réutilisable par `eval/compare_models.py`
pour produire les variantes "RAG-only", "DPO+RAG", "RLHF+RAG".

Usage standalone (sanity check) :
  python rag/inference.py --query "Is it wrong to lie to spare someone's feelings?"
"""

import argparse
import json
import os
import pickle
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EthicsRetriever:
    def __init__(self, index_dir: str, embedder_name: Optional[str] = None):
        self.index_dir = index_dir
        with open(os.path.join(index_dir, "meta.json")) as f:
            meta = json.load(f)
        self.embedder_name = embedder_name or meta["embedder"]
        print(f"[RAG] Loading embedder: {self.embedder_name}")
        self.embedder = SentenceTransformer(self.embedder_name)
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "docs.pkl"), "rb") as f:
            self.docs = pickle.load(f)
        print(f"[RAG] Index loaded ({self.index.ntotal} docs, dim={self.index.d})")

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self.index.search(q_emb, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            doc = dict(self.docs[idx])
            doc["score"] = float(score)
            out.append(doc)
        return out

    def augment_prompt(self, original_prompt: str, k: int = 3) -> str:
        """Insère les principes récupérés EN AMONT du prompt original."""
        retrieved = self.retrieve(original_prompt, k=k)
        if not retrieved:
            return original_prompt
        principles_block = "\n".join(
            f"- ({d['framework']}) {d['text']}" for d in retrieved
        )
        return (
            "You are answering an ethics question. Use the following relevant ethical "
            "principles as guidance:\n"
            f"{principles_block}\n\n"
            "Now answer the question below using these principles.\n\n"
            f"{original_prompt}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="rag/corpus")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    retriever = EthicsRetriever(args.index_dir)
    results = retriever.retrieve(args.query, k=args.k)
    print(f"\nTop-{args.k} retrieved for: {args.query!r}\n")
    for r in results:
        print(f"  [{r['score']:.3f}] ({r['framework']}) {r['title']}")
        print(f"     {r['text'][:140]}…\n")

    print("─── Augmented prompt ───")
    print(retriever.augment_prompt(args.query, k=args.k))
