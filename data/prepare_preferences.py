"""
data/prepare_preferences.py
───────────────────────────
Construit le jeu hybride de paires de préférences pour DPO/RM/PPO.

Composition :
  1. PKU-SafeRLHF       — préférences humaines safety-oriented
  2. UltraFeedback      — préférences générales haute qualité
  3. Synthétique éthique — (optionnel) format aligné sur ETHICS (généré séparément)

Format de sortie (JSONL) :
  {"prompt": str, "chosen": str, "rejected": str, "source": "pku|ultra|synth"}

Le prompt est formaté plus tard avec le chat template du tokenizer dans les scripts
d'entraînement, donc on garde ici les textes bruts.

Usage:
  python data/prepare_preferences.py --n_pku 10000 --n_ultra 5000 --out_path data/preferences.jsonl
  python data/prepare_preferences.py --include_synthetic data/synthetic_ethics.jsonl
"""

import argparse
import json
import os
import random
from typing import Iterable

from datasets import load_dataset


def write_jsonl(path: str, rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ── Source 1 : PKU-SafeRLHF ───────────────────────────────────────────────────
def load_pku_pairs(n_samples: int) -> list[dict]:
    """
    PKU-SafeRLHF a deux dimensions (better_response_id et safer_response_id).
    On utilise safer_response_id pour orienter vers la safety/ethics.
    """
    print(f"[PKU] Loading PKU-Alignment/PKU-SafeRLHF (target n={n_samples})…")
    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    except Exception as e:
        print(f"[PKU] Failed: {e}")
        return []

    rows = []
    for ex in ds:
        # Format PKU : prompt, response_0, response_1, safer_response_id (0/1)
        safer_id = ex.get("safer_response_id")
        if safer_id not in (0, 1):
            continue
        prompt = ex.get("prompt", "").strip()
        r0 = ex.get("response_0", "").strip()
        r1 = ex.get("response_1", "").strip()
        if not prompt or not r0 or not r1 or r0 == r1:
            continue
        chosen = r0 if safer_id == 0 else r1
        rejected = r1 if safer_id == 0 else r0
        rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "pku",
        })
        if len(rows) >= n_samples:
            break
    print(f"[PKU] -> {len(rows)} pairs")
    return rows


# ── Source 2 : UltraFeedback ──────────────────────────────────────────────────
def load_ultrafeedback_pairs(n_samples: int) -> list[dict]:
    """
    UltraFeedback (binarized) — paires chosen/rejected déjà formées.
    """
    print(f"[Ultra] Loading argilla/ultrafeedback-binarized-preferences (target n={n_samples})…")
    try:
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")
    except Exception:
        try:
            ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        except Exception as e:
            print(f"[Ultra] Failed: {e}")
            return []

    rows = []
    for ex in ds:
        # Plusieurs formats possibles selon la version
        if "chosen" in ex and "rejected" in ex:
            chosen = ex["chosen"]
            rejected = ex["rejected"]
            # Si format chat (list[dict]) on prend le dernier message assistant
            if isinstance(chosen, list):
                # extract prompt from user msgs and last assistant content
                user_msgs = [m["content"] for m in chosen if m.get("role") == "user"]
                prompt = user_msgs[-1] if user_msgs else ""
                chosen_text = next((m["content"] for m in reversed(chosen) if m.get("role") == "assistant"), "")
                rejected_text = next((m["content"] for m in reversed(rejected) if m.get("role") == "assistant"), "")
            else:
                prompt = ex.get("prompt", "")
                chosen_text = chosen
                rejected_text = rejected
        elif "chosen_response" in ex:
            prompt = ex.get("instruction") or ex.get("prompt", "")
            chosen_text = ex["chosen_response"]
            rejected_text = ex["rejected_response"]
        else:
            continue

        prompt = (prompt or "").strip()
        chosen_text = (chosen_text or "").strip()
        rejected_text = (rejected_text or "").strip()
        if not prompt or not chosen_text or not rejected_text or chosen_text == rejected_text:
            continue
        rows.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "source": "ultra",
        })
        if len(rows) >= n_samples:
            break
    print(f"[Ultra] -> {len(rows)} pairs")
    return rows


# ── Source 3 : Synthétique éthique (depuis un JSONL externe) ─────────────────
def load_synthetic_pairs(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        print(f"[Synth] No file at {path}, skipping.")
        return []
    print(f"[Synth] Reading {path}…")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in ex for k in ("prompt", "chosen", "rejected")):
                continue
            ex["source"] = "synth"
            rows.append(ex)
    print(f"[Synth] -> {len(rows)} pairs")
    return rows


# ── Filtrage qualité ──────────────────────────────────────────────────────────
def filter_pair(p: dict, min_len: int = 5, max_len: int = 2000) -> bool:
    for k in ("prompt", "chosen", "rejected"):
        text = p.get(k, "")
        if not isinstance(text, str):
            return False
        n = len(text)
        if n < min_len or n > max_len:
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pku", type=int, default=10000)
    parser.add_argument("--n_ultra", type=int, default=5000)
    parser.add_argument("--include_synthetic", type=str, default=None,
                        help="Path to a synthetic JSONL produced by generate_synthetic.py")
    parser.add_argument("--out_path", type=str, default="data/preferences.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_rows: list[dict] = []
    all_rows.extend(load_pku_pairs(args.n_pku))
    all_rows.extend(load_ultrafeedback_pairs(args.n_ultra))
    if args.include_synthetic:
        all_rows.extend(load_synthetic_pairs(args.include_synthetic))

    before = len(all_rows)
    all_rows = [r for r in all_rows if filter_pair(r)]
    print(f"Filtered {before} -> {len(all_rows)} pairs")

    random.shuffle(all_rows)
    n = write_jsonl(args.out_path, all_rows)
    print(f"Wrote {n} pairs to {args.out_path}")
    # Petit récap par source
    by_source = {}
    for r in all_rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for k, v in by_source.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
