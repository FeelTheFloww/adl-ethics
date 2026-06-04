"""Significativité statistique des écarts ETHICS (sans réentraînement).

Lit le JSON produit par evaluate_ethics_fixed.py (qui sauvegarde la correction
par exemple dans `preds`, alignée entre conditions) et calcule :

  * Wilson 95 % CI pour l'accuracy globale de chaque condition ;
  * test de McNemar apparié (baseline vs chaque méthode) sur les 500 exemples ;
  * bootstrap apparié de l'écart d'accuracy (CI à 95 %).

C'est la pièce qui transforme "les écarts sont sous la variance" d'une intuition
en un résultat : les exemples d'éval sont fixes et identiques entre conditions,
donc McNemar est le test adapté.

Usage : python eval/stats_analysis.py --results results/eval_results_fixed.json
"""

from __future__ import annotations

import argparse
import json
import math
import random

CATEGORIES = ["commonsense", "deontology", "justice", "virtue", "utilitarianism"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def concat_preds(cond: dict) -> list[int]:
    """Concatène les vecteurs `preds` (1=correct) sur les catégories."""
    out = []
    for cat in CATEGORIES:
        out.extend(cond.get(cat, {}).get("preds", []))
    return out


def mcnemar_exact(a_correct: list[int], b_correct: list[int]) -> dict:
    """McNemar exact (binomial) sur des prédictions appariées (1=correct)."""
    n = min(len(a_correct), len(b_correct))
    b = c = 0  # b : A correct & B faux ; c : A faux & B correct
    for i in range(n):
        if a_correct[i] == 1 and b_correct[i] == 0:
            b += 1
        elif a_correct[i] == 0 and b_correct[i] == 1:
            c += 1
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "discordant": 0, "p_value": 1.0}
    k = min(b, c)
    # p exacte bilatérale = 2 * P(X <= k) sous Binom(nd, 0.5)
    cum = sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd)
    return {"b": b, "c": c, "discordant": nd, "p_value": min(1.0, 2 * cum)}


def paired_bootstrap(a: list[int], b: list[int], iters=10000, seed=0) -> tuple[float, float, float]:
    """CI 95 % de l'écart d'accuracy (B - A) par bootstrap apparié."""
    n = min(len(a), len(b))
    rng = random.Random(seed)
    diffs = []
    base = sum(b[:n]) / n - sum(a[:n]) / n
    for _ in range(iters):
        s = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += b[i] - a[i]
        diffs.append(s / n)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    return base, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/eval_results_fixed.json")
    ap.add_argument("--baseline_key", default="baseline")
    ap.add_argument("--bootstrap_iters", type=int, default=10000)
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    base_preds = concat_preds(data[args.baseline_key])
    n = len(base_preds)
    if n == 0:
        print("Aucune prédiction trouvée — l'éval n'a produit aucun exemple "
              "(vérifie le chargement d'ETHICS dans evaluate_ethics_fixed.py).")
        return
    k = sum(base_preds)
    lo, hi = wilson_ci(k, n)
    print(f"\nBaseline accuracy = {k}/{n} = {k/n:.3f}  "
          f"(Wilson 95% CI [{lo:.3f}, {hi:.3f}])\n")
    print(f"{'Condition':<12}{'acc':>7}{'Δ vs base':>11}{'boot 95% CI':>20}{'McNemar p':>12}")
    print("-" * 62)

    for cond, res in data.items():
        preds = concat_preds(res)
        if not preds:
            continue
        kk, nn = sum(preds), len(preds)
        if cond == args.baseline_key:
            print(f"{cond:<12}{kk/nn:>7.3f}{'—':>11}{'—':>20}{'—':>12}")
            continue
        mc = mcnemar_exact(base_preds, preds)
        delta, blo, bhi = paired_bootstrap(base_preds, preds, args.bootstrap_iters)
        print(f"{cond:<12}{kk/nn:>7.3f}{delta:>+11.3f}"
              f"   [{blo:+.3f}, {bhi:+.3f}]{mc['p_value']:>12.3f}")

    print("\nLecture : p > 0,05 ⇒ écart non significatif ; un CI bootstrap qui "
          "contient 0 confirme l'absence d'effet directionnel.")

    # ------- McNemar par sous-ensemble (chaque méthode vs baseline) -------
    print("\n" + "=" * 64)
    print("McNemar par sous-ensemble (vs baseline)")
    print("=" * 64)
    methods = [c for c in data if c != args.baseline_key]
    header = f"{'Sous-ensemble':<15}{'base':>7}" + "".join(
        f"{m[:7]:>10}{'p':>7}" for m in methods)
    print(header)
    print("-" * len(header))
    for cat in CATEGORIES:
        bp = data[args.baseline_key].get(cat, {}).get("preds", [])
        if not bp:
            continue
        base_acc = sum(bp) / len(bp)
        row = f"{cat:<15}{base_acc:>7.3f}"
        for m in methods:
            mp = data[m].get(cat, {}).get("preds", [])
            if not mp:
                row += f"{'—':>10}{'—':>7}"
                continue
            acc = sum(mp) / len(mp)
            p = mcnemar_exact(bp, mp)["p_value"]
            star = "*" if p < 0.05 else " "
            row += f"{acc:>9.3f}{star}{p:>7.3f}"
        print(row)
    print("\n* = significatif à p < 0,05 (test apparié, même 100 exemples par "
          "sous-ensemble entre conditions).")


if __name__ == "__main__":
    main()
