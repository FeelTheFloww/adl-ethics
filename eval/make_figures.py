"""Génère les figures du rapport corrigé à partir des résultats réels.

Entrées :
  - results/eval_results_fixed.json (sortie de evaluate_ethics_fixed.py)
  - results/kl_diagnostics.json (sortie de kl_diagnostics.py) [optionnel ;
    valeurs par défaut = celles mesurées sur notre run]

Sorties (figures/) :
  - fig_kl_diagnostic.pdf : KL à l'init (3 configs) vs KL observée en entraînement
  - fig_subset_majority.pdf : accuracy par sous-ensemble vs classe majoritaire
  - fig_delta_forest.pdf : écart au baseline + IC bootstrap 95 % (McNemar)

Usage : python eval/make_figures.py
"""
from __future__ import annotations
import argparse, json, math, os, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CATS = ["commonsense", "deontology", "justice", "virtue", "utilitarianism"]
CATS_FR = ["Sens-commun", "Déontologie", "Justice", "Vertu", "Utilitarisme"]
CONDS = ["baseline", "dpo", "rlhf", "rag"]
CONDS_LBL = {"baseline": "Baseline", "dpo": "DPO", "rlhf": "RLHF", "rag": "RAG"}
COL = {"baseline": "#4C6EF5", "dpo": "#12B886", "rlhf": "#F59F00", "rag": "#E8590C"}

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "savefig.bbox": "tight", "figure.dpi": 150,
})


def concat_preds(cond):
    out = []
    for c in CATS:
        out.extend(cond.get(c, {}).get("preds", []))
    return out


def mcnemar_boot(a, b, iters=20000, seed=0):
    n = min(len(a), len(b))
    rng = random.Random(seed)
    base = sum(b[:n]) / n - sum(a[:n]) / n
    diffs = []
    for _ in range(iters):
        s = sum(b[i] - a[i] for i in (rng.randrange(n) for _ in range(n)))
        diffs.append(s / n)
    diffs.sort()
    return base, diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


# --------------------------------------------------------------------------- #
def fig_kl(kl, out):
    """Barres : KL d'initialisation (3 configs) ; ligne = KL en entraînement."""
    labels = ["Config initiale\n(asymétrique)", "Symétrique", "Référence bf16"]
    vals = [abs(kl["A_buggy"]["kl_sum"]), abs(kl["B_fixed_symmetric"]["kl_sum"]),
            abs(kl["C_bf16_ref"]["kl_sum"])]
    train_kl = 110.0

    fig, (axt, axb) = plt.subplots(2, 1, sharex=True, figsize=(3.3, 3.2),
                                   gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.08})
    x = range(len(vals))
    for ax in (axt, axb):
        ax.bar(x, vals, color="#4C6EF5", width=0.6, zorder=3)
        ax.axhline(train_kl, color="#E03131", ls="--", lw=1.4, zorder=2)
    # panneau haut = zone ~110 ; bas = zone ~0-7
    axt.set_ylim(100, 122); axb.set_ylim(0, 7.6)
    axt.text(len(vals) - 1, train_kl + 1.5, "KL observée en entraînement ≈ 110",
             ha="right", va="bottom", color="#E03131", fontsize=8)
    # barres de rupture d'axe
    axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
    axt.tick_params(labelbottom=False, bottom=False)
    d = .012
    kw = dict(transform=axt.transAxes, color="k", clip_on=False, lw=0.8)
    axt.plot((-d, +d), (-d*2, +d*2), **kw); axt.plot((1-d, 1+d), (-d*2, +d*2), **kw)
    kw["transform"] = axb.transAxes
    axb.plot((-d, +d), (1-d, 1+d), **kw); axb.plot((1-d, 1+d), (1-d, 1+d), **kw)
    for i, v in enumerate(vals):
        axb.text(i, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axb.set_xticks(list(x)); axb.set_xticklabels(labels, fontsize=8)
    axb.set_ylabel("Divergence KL (somme/réponse)")
    axt.set_title("KL à l'initialisation vs en entraînement", fontsize=9)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("->", out)


def fig_subsets(data, out):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    n = len(CONDS); w = 0.18
    for j, cond in enumerate(CONDS):
        accs = [data[cond][c]["accuracy"] for c in CATS]
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(CATS))]
        ax.bar(xs, accs, width=w, label=CONDS_LBL[cond], color=COL[cond], zorder=3)
    # classe majoritaire : trait horizontal par sous-ensemble
    for i, c in enumerate(CATS):
        mj = data["baseline"][c].get("majority_baseline")
        if mj is None:
            continue
        ax.plot([i - 0.45, i + 0.45], [mj, mj], color="#212529", lw=1.6, ls=(0, (4, 2)),
                zorder=5)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", zorder=1)
    ax.text(len(CATS) - 0.5, 0.505, "hasard", color="grey", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS_FR)
    ax.set_ylim(0.35, 0.85); ax.set_ylabel("Précision")
    ax.set_title("Précision par sous-ensemble ETHICS (— = classe majoritaire)", fontsize=9)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("->", out)


def fig_forest(data, out):
    base = concat_preds(data["baseline"])
    rows = []
    for cond in ["dpo", "rlhf", "rag"]:
        delta, lo, hi = mcnemar_boot(base, concat_preds(data[cond]))
        rows.append((CONDS_LBL[cond], delta, lo, hi, COL[cond]))
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for y, (lbl, d, lo, hi, c) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=3)
        ax.plot(d, y, "o", color=c, ms=6, zorder=4)
        ax.text(hi + 0.004, y, f"{d:+.3f}", va="center", fontsize=8, color=c)
    ax.axvline(0, color="#212529", lw=1.0, ls="--", zorder=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Écart d'accuracy globale vs baseline")
    ax.set_title("Écart au baseline (IC bootstrap 95 %)", fontsize=9)
    ax.margins(x=0.18)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("->", out)


def fig_confusion(data, out, category="commonsense", cond="baseline"):
    """Matrice de confusion 2x2 pour un sous-ensemble binaire (si labels bruts présents)."""
    blk = data.get(cond, {}).get(category, {})
    gold, pred = blk.get("gold"), blk.get("pred")
    if not gold or not pred:
        print(f"  [confusion] pas de labels bruts pour {cond}/{category} — "
              "relance evaluate_ethics_fixed.py (version qui sauvegarde gold/pred).")
        return
    cm = [[0, 0], [0, 0]]   # cm[gold][pred]
    for g, p in zip(gold, pred):
        cm[int(g)][int(p)] += 1
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    labels01 = ["0\n(non/acceptable)", "1\n(oui/répréhensible)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité terrain")
    total = sum(sum(r) for r in cm)
    for i in range(2):
        for j in range(2):
            v = cm[i][j]
            ax.text(j, i, f"{v}\n({100*v/total:.0f}\\%)", ha="center", va="center",
                    color="white" if v > total*0.25 else "black", fontsize=9)
    acc = (cm[0][0]+cm[1][1])/total
    ax.set_title(f"Matrice de confusion — {category}\n({CONDS_LBL.get(cond,cond)}, "
                 f"acc={acc:.2f})", fontsize=9)
    ax.grid(False)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/eval_results_fixed.json")
    ap.add_argument("--kl", default="results/kl_diagnostics.json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    data = json.load(open(args.results, encoding="utf-8"))
    # valeurs de repli = celles mesurées sur notre run
    kl = {"A_buggy": {"kl_sum": -0.12}, "B_fixed_symmetric": {"kl_sum": 0.0},
          "C_bf16_ref": {"kl_sum": 6.27}}
    if os.path.isfile(args.kl):
        kl = json.load(open(args.kl, encoding="utf-8"))
    fig_kl(kl, os.path.join(args.outdir, "fig_kl_diagnostic.pdf"))
    fig_subsets(data, os.path.join(args.outdir, "fig_subset_majority.pdf"))
    fig_forest(data, os.path.join(args.outdir, "fig_delta_forest.pdf"))
    fig_confusion(data, os.path.join(args.outdir, "fig_confusion_commonsense.pdf"),
                  category="commonsense", cond="baseline")


if __name__ == "__main__":
    main()
