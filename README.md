# ADL Project — Ethical Alignment of Small Language Models

Comparaison de **trois stratégies d'alignement** sur le benchmark ETHICS (Hendrycks et al., 2021), à partir d'un modèle de base Qwen2.5-1.5B-Instruct.

Méthodes implémentées :
- **DPO** (Direct Preference Optimization)
- **RLHF** (Reward Model + PPO)
- **RAG-based alignment** (récupération + injection de principes éthiques)

Le rapport ACL final met l'accent sur la comparaison **DPO vs RLHF** (même données, deux paradigmes d'optimisation). RAG est utilisé comme module additionnel inférence-time, présent dans la table de résultats et l'ablation.

## Structure

```
ADL/
├── setup.sh                          # Installation des dépendances
├── data/
│   ├── prepare_preferences.py        # Construit le jeu hybride de préférences
│   ├── generate_synthetic.py         # (Optionnel) Génère des paires synthétiques via API
│   └── ethical_corpus.json           # Corpus de principes éthiques (seed RAG)
├── training/
│   ├── train_dpo.py                  # DPO QLoRA sur données hybrides
│   ├── train_reward_model.py         # Reward model (RLHF étape 1)
│   └── train_ppo.py                  # PPO policy optimization (RLHF étape 2)
├── rag/
│   ├── build_index.py                # FAISS index depuis ethical_corpus.json
│   ├── inference.py                  # Retrieval + augmentation de prompt
│   └── corpus/                       # Index FAISS sauvegardé
├── eval/
│   ├── evaluate_baseline.py          # Évalue un modèle sur ETHICS (5 subsets)
│   └── compare_models.py             # Compare toutes les variantes (table + plot)
├── configs/                          # Configs YAML (optionnel)
├── results/                          # Modèles, scores JSON, graphes
└── report/                           # Rapport ACL (LaTeX)
```

## Stratégie de données

Pour l'entraînement on construit un **mélange hybride** (script `data/prepare_preferences.py`) :

1. **PKU-SafeRLHF** (subset ~10K paires) — préférences humaines réelles, orientées safety/harmfulness.
2. **UltraFeedback** (subset ~5K paires) — préférences générales haute qualité (4 dimensions agrégées).
3. **Synthétique format-aligné** (~3–5K paires, *optionnel*) — généré via API (Claude/GPT) pour produire des paires sous le format de classification morale qui ressemble à ETHICS, sans jamais voir ETHICS-train.

L'évaluation utilise **uniquement le split test d'ETHICS** (100 exemples par subset : commonsense, justice, deontology, virtue, utilitarianism).

## Ordre d'exécution

```bash
# 1. Setup
bash setup.sh
source venv/bin/activate
wandb login

# 2. Baseline (toujours en premier — référence)
python eval/evaluate_baseline.py --model_name Qwen/Qwen2.5-1.5B-Instruct

# 3. Préparation des données de préférence
python data/prepare_preferences.py --out_path data/preferences.jsonl --n_pku 10000 --n_ultra 5000

# 3b. (Optionnel) Synthétique — nécessite ANTHROPIC_API_KEY ou OPENAI_API_KEY
python data/generate_synthetic.py --n 3000 --out_path data/synthetic_ethics.jsonl
python data/prepare_preferences.py --include_synthetic data/synthetic_ethics.jsonl

# 4. DPO
python training/train_dpo.py --data_path data/preferences.jsonl

# 5. RLHF — étape 1 : reward model
python training/train_reward_model.py --data_path data/preferences.jsonl

# 5b. RLHF — étape 2 : PPO
python training/train_ppo.py --reward_model_path results/reward_model

# 6. RAG — index
python rag/build_index.py --corpus data/ethical_corpus.json

# 7. Comparaison finale (baseline / DPO / RLHF / RAG / DPO+RAG / RLHF+RAG)
python eval/compare_models.py
```

## Hyperparamètres clés

| Hyperparamètre        | DPO        | Reward Model | PPO        |
|---|---|---|---|
| LoRA rank             | 16         | 16           | 16         |
| LoRA target modules   | q,k,v,o    | q,k,v,o      | q,k,v,o    |
| Batch size effectif   | 16         | 16           | 8          |
| Learning rate         | 5e-5       | 2e-5         | 1e-5       |
| Beta (DPO)            | 0.1        | —            | —          |
| KL coef (PPO)         | —          | —            | 0.1        |
| Max length            | 512        | 512          | 256        |
| Epochs                | 1          | 1            | 1 epoch PPO|

## Contraintes machine

- RTX 4060 8 GB VRAM → QLoRA 4-bit obligatoire partout.
- PPO peut être déplacé sur Kaggle T4×2 si OOM en local (commenter `device_map="auto"` et adapter).

## Notes importantes

- **Ne jamais utiliser ETHICS pour l'entraînement** (consigne du sujet).
- Logger tous les runs sur W&B → récupérer les courbes pour le rapport.
- Sauvegarder les résultats JSON à chaque étape (`results/*.json`).
- Le synthétique passe par une API externe → documenter dans le rapport (« external LLMs used ».)
