"""
eval/compare_models.py
──────────────────────
Compare toutes les variantes sur ETHICS et produit la table + le graphe pour le
rapport ACL.

Variantes supportées :
  - Baseline           : Qwen2.5-1.5B-Instruct brut
  - DPO                : base + adapter DPO
  - RLHF               : base + adapter PPO
  - RAG-only           : base + retrieval (pas d'entraînement)
  - DPO+RAG            : DPO + retrieval
  - RLHF+RAG           : RLHF + retrieval

Pour le rapport (8 pages ACL), le récit principal est DPO vs RLHF. Les variantes
RAG figurent en ablation.

Usage :
  python eval/compare_models.py
"""

import json
import os
import sys
import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Permet l'import depuis eval/ vers eval/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate_baseline import (
    ETHICS_SUBSETS, N_EXAMPLES, PROMPTS,
    load_model, load_ethics_subset, extract_text_and_label,
    predict_classification, predict_utilitarianism,
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Modèles à évaluer. Les adapters sont skippés s'ils n'existent pas
# (utile pour tester avant que tout soit entraîné).
MODELS = {
    "Baseline":   {"peft": None,                       "rag": False},
    "DPO":        {"peft": "results/dpo_model",        "rag": False},
    "RLHF":       {"peft": "results/rlhf_model",       "rag": False},
    "RAG-only":   {"peft": None,                       "rag": True},
    "DPO+RAG":    {"peft": "results/dpo_model",        "rag": True},
    "RLHF+RAG":   {"peft": "results/rlhf_model",       "rag": True},
}


# ── Évaluation d'un modèle, avec ou sans RAG ──────────────────────────────────
def evaluate_with_optional_rag(model, tokenizer, subset: str, retriever=None) -> dict:
    """Évalue le modèle sur un subset ; si retriever fourni, augmente le prompt."""
    print(f"\n[{subset}] Loading…")
    ds = load_ethics_subset(subset)
    ds = ds.select(range(min(N_EXAMPLES, len(ds))))

    correct, total = 0, 0
    errors = []

    if subset == "utilitarianism":
        # Le RAG n'est pas appliqué ici (scoring de log-likelihood pur, pas de prompt)
        from tqdm import tqdm
        for ex in tqdm(ds, desc=subset):
            baseline_text = ex.get("baseline", "")
            less_text = ex.get("less_pleasant", "")
            if not baseline_text or not less_text:
                continue
            pred = predict_utilitarianism(model, tokenizer, baseline_text, less_text)
            if pred == 1:
                correct += 1
            else:
                errors.append({"baseline": baseline_text[:80], "less_pleasant": less_text[:80]})
            total += 1
    else:
        from tqdm import tqdm
        for ex in tqdm(ds, desc=subset):
            text, label = extract_text_and_label(ex, subset)
            if not text:
                continue
            base_prompt = PROMPTS[subset].format(text=text)
            prompt = retriever.augment_prompt(base_prompt, k=3) if retriever is not None else base_prompt
            pred = predict_classification(model, tokenizer, prompt)
            if pred == label:
                correct += 1
            else:
                errors.append({"text": text[:120], "pred": pred, "label": label})
            total += 1

    acc = correct / total if total > 0 else 0.0
    return {"subset": subset, "accuracy": acc, "correct": correct, "total": total,
            "sample_errors": errors[:5]}


# ── Charge le retriever une fois pour réutiliser ─────────────────────────────
def maybe_load_retriever(index_dir: str = "rag/corpus"):
    if not Path(index_dir).exists() or not (Path(index_dir) / "index.faiss").exists():
        print(f"[!] No RAG index found at {index_dir} — RAG variants will be skipped.")
        return None
    from rag.inference import EthicsRetriever
    return EthicsRetriever(index_dir)


# ── Plotting / table ──────────────────────────────────────────────────────────
def print_summary_table(all_results: dict):
    print("\n" + "=" * 80)
    header = f"{'Model':<14} | " + " | ".join(f"{s[:9]:<9}" for s in ETHICS_SUBSETS) + " | Overall"
    print(header)
    print("-" * len(header))
    for model_name, results in all_results.items():
        accs = []
        for s in ETHICS_SUBSETS:
            match = next((r["accuracy"] for r in results["results_by_subset"] if r["subset"] == s), 0.0)
            accs.append(match)
        overall = results["overall_accuracy"]
        row = f"{model_name:<14} | " + " | ".join(f"{a:<9.3f}" for a in accs) + f" | {overall:.3f}"
        print(row)
    print("=" * 80)


def plot_results(all_results: dict, output_path: str = "results/comparison.png"):
    subsets = ETHICS_SUBSETS
    models = list(all_results.keys())
    n_models = len(models)
    x = np.arange(len(subsets))
    width = 0.8 / max(n_models, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model_name, results) in enumerate(all_results.items()):
        accs = [
            next((r["accuracy"] for r in results["results_by_subset"] if r["subset"] == s), 0.0)
            for s in subsets
        ]
        ax.bar(x + i * width, accs, width, label=model_name, color=colors[i], alpha=0.9)

    ax.set_xlabel("ETHICS Subset")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ethical Alignment on ETHICS — All Variants")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([s.capitalize() for s in subsets])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    retriever = maybe_load_retriever()

    all_results = {}
    for model_name, config in MODELS.items():
        # Skip si l'adapter n'existe pas encore
        if config["peft"] is not None and not Path(config["peft"]).exists():
            print(f"[skip] {model_name}: adapter not found at {config['peft']}")
            continue
        # Skip RAG si pas d'index
        if config["rag"] and retriever is None:
            print(f"[skip] {model_name}: RAG index not built")
            continue

        print(f"\n==== {model_name} ====")
        model, tokenizer = load_model(BASE_MODEL, peft_path=config["peft"])
        results_by_subset = []
        for subset in ETHICS_SUBSETS:
            try:
                res = evaluate_with_optional_rag(model, tokenizer, subset,
                                                 retriever if config["rag"] else None)
            except Exception as e:
                print(f"  [{subset}] FAILED: {e}")
                res = {"subset": subset, "accuracy": 0.0, "correct": 0, "total": 0,
                       "error": str(e), "sample_errors": []}
            results_by_subset.append(res)

        total_correct = sum(r["correct"] for r in results_by_subset)
        total_examples = sum(r["total"] for r in results_by_subset)
        overall = total_correct / total_examples if total_examples > 0 else 0.0
        all_results[model_name] = {
            "overall_accuracy": overall,
            "results_by_subset": results_by_subset,
        }
        print(f"  -> overall = {overall:.3f}")

        # Libère VRAM avant le prochain modèle
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Sauvegarde
    out_path = "results/comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")

    print_summary_table(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
