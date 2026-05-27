"""Trace l'évolution de la divergence KL durant l'entraînement PPO, à partir du
trainer_state.json produit par TRL. Génère la figure utilisée dans le rapport.

Usage : python eval/plot_ppo_kl.py
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_state", type=str,
                        default="results/rlhf_model/checkpoint-125/trainer_state.json")
    parser.add_argument("--out", type=str, default="results/ppo_kl_divergence.png")
    parser.add_argument("--target_kl", type=float, default=1.0,
                        help="Référence de KL pour un PPO stable.")
    args = parser.parse_args()

    with open(args.trainer_state, encoding="utf-8") as f:
        state = json.load(f)

    logs = [e for e in state["log_history"] if "objective/kl" in e]
    steps = [e["step"] for e in logs]
    kl = [e["objective/kl"] for e in logs]
    mean_kl = sum(kl) / len(kl)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(steps, kl, color="#E24B4A", alpha=0.08)
    ax.plot(steps, kl, color="#E24B4A", linewidth=1.6, label="KL par étape")
    ax.axhline(mean_kl, color="#888780", linestyle="--", linewidth=1.3,
               label=f"Moyenne ≈ {mean_kl:.0f}")
    ax.axhline(args.target_kl, color="#1D9E75", linestyle=":", linewidth=1.4,
               label=f"Cible PPO stable ≲ {args.target_kl:g}")

    ax.set_xlabel("Étape PPO (gradient step)")
    ax.set_ylabel("Divergence KL")
    ax.set_title("Divergence KL durant l'entraînement PPO (RLHF)")
    ax.set_ylim(0, max(kl) * 1.1)
    ax.set_xlim(min(steps), max(steps))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    pdf_path = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_path)
    print(f"Saved -> {args.out} et {pdf_path}")
    print(f"KL moyenne = {mean_kl:.2f}  (min {min(kl):.1f}, max {max(kl):.1f}, n={len(kl)})")


if __name__ == "__main__":
    main()
