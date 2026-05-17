# ADL Project — Ethical Alignment of Small Language Models

Comparaison de **trois stratégies d'alignement** sur le benchmark ETHICS (Hendrycks et al., 2021), avec Qwen2.5-1.5B-Instruct comme modèle de base.

Méthodes :
- **DPO** (Direct Preference Optimization)
- **RLHF** (Reward Model + PPO)
- **RAG-based alignment** (retrieval + injection de principes éthiques)

## Structure

```
ADL/
├── setup.sh                            # Installation Linux/Kaggle
├── data/
│   ├── prepare_preferences.py          # Construit le jeu hybride PKU + UltraFeedback (+ synthétique optionnel)
│   ├── generate_synthetic.py           # Génère des paires synthétiques (Groq/Anthropic/OpenAI)
│   ├── ethical_corpus.json             # Corpus RAG v1 (21 docs)
│   └── ethical_corpus_v2.json          # Corpus RAG v2 (170 docs — recommandé)
├── training/
│   ├── train_dpo.py                    # DPO LoRA bf16
│   ├── train_reward_model.py           # Reward model (RLHF étape 1)
│   └── train_ppo.py                    # PPO TRL 0.12 + bnb 4-bit (RLHF étape 2)
├── rag/
│   ├── build_index.py                  # FAISS index depuis ethical_corpus
│   ├── inference.py                    # EthicsRetriever (retrieval + prompt augmentation)
│   └── corpus/                         # Index FAISS sauvegardé
├── eval/
│   ├── evaluate_baseline.py            # Évalue un modèle sur ETHICS (5 subsets)
│   └── compare_models.py               # Compare toutes les variantes (table + plot)
├── results/                            # Adapters, scores JSON, graphes
├── report/
│   ├── main.tex                        # Squelette papier ACL pré-rempli
│   └── references.bib                  # Bibliographie BibTeX
└── cv/                                 # CVs personnels (non publics par défaut)
```

## Environnement supporté

| Plateforme | Verdict | Notes |
|---|---|---|
| **Kaggle T4×2 (16 GB×2)** | ✅ Recommandé | Set `CUDA_VISIBLE_DEVICES=0` pour éviter le sharding |
| Linux + GPU consumer ≥ 8 GB | ✅ OK | bnb 4-bit + bf16, Python 3.11 |
| WSL Ubuntu | ✅ OK | Idem Linux |
| Windows natif | ⚠️ Partiel | DPO/RM/eval OK, **PPO crashe** (segfault bnb optimizers sous Windows) |

**Python conseillé** : 3.11 (3.12 OK, 3.13 a des trous de wheels pour torch CUDA et bnb).

## Workflow Kaggle (recommandé pour le PPO)

### Setup notebook
1. New Notebook → Settings : **Accelerator = GPU T4 x1**, **Internet = On**
2. Première cellule **avant tout import** :
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"
   ```
3. Add-Ons → Secrets → ajouter `ANTHROPIC_API_KEY` ou `GROQ_API_KEY` si tu utilises le synthétique
4. Cellule install :
   ```python
   !pip install -q -U bitsandbytes transformers==4.46.3 trl==0.12.0 peft==0.14.0 \
       accelerate==1.2.0 datasets==3.2.0 sentence-transformers faiss-cpu \
       anthropic openai pyarrow==17.0.0 tqdm
   ```
   Puis **Run → Restart Session** (obligatoire pour que les libs soient rechargées).
5. Clone le repo :
   ```python
   !git clone https://github.com/FeelTheFloww/adl-ethics.git /kaggle/working/adl
   %cd /kaggle/working/adl
   ```

### Pipeline complet

```bash
# 1. Baseline
python eval/evaluate_baseline.py --model_name Qwen/Qwen2.5-1.5B-Instruct

# 2. Prep données (PKU + Ultra ; ajouter --include_synthetic si tu as généré le synthétique)
python data/prepare_preferences.py --n_pku 15000 --n_ultra 5000 --out_path data/preferences.jsonl

# 3. (Optionnel) Synthétique via API
python data/generate_synthetic.py --provider anthropic --n 3000 --out_path data/synthetic_ethics.jsonl
python data/prepare_preferences.py --n_pku 15000 --n_ultra 5000 \
    --include_synthetic data/synthetic_ethics.jsonl --out_path data/preferences.jsonl

# 4. RAG index (utilise corpus v2 par défaut)
python rag/build_index.py --corpus data/ethical_corpus_v2.json --out_dir rag/corpus

# 5. DPO
python training/train_dpo.py --data_path data/preferences.jsonl --batch_size 4 --grad_accum 4

# 6. Reward Model
python training/train_reward_model.py --data_path data/preferences.jsonl --batch_size 4 --grad_accum 4

# 7. PPO (Kaggle uniquement, bnb 4-bit)
python training/train_ppo.py \
    --dpo_adapter results/dpo_model \
    --reward_model_path results/reward_model \
    --data_path data/preferences.jsonl \
    --output_dir results/rlhf_model

# 8. Comparaison finale (6 variantes)
python eval/compare_models.py
```

## Stratégie de données

| Source | Taille | Pourquoi |
|---|---|---|
| PKU-SafeRLHF | ~15K paires | Préférences humaines orientées safety/harmlessness |
| UltraFeedback | ~5K paires | Préférences générales haute qualité |
| Synthétique (optionnel) | ~3K paires | Format-aligné sur ETHICS pour combler le gap dialogue ↔ classification |

L'évaluation utilise **uniquement le test split d'ETHICS** (100 exemples × 5 subsets : commonsense, justice, deontology, virtue, utilitarianism).

## Hyperparamètres clés

| Param | DPO | Reward Model | PPO |
|---|---|---|---|
| LoRA rank | 16 | 16 | 16 |
| Target modules | q,k,v,o | q,k,v,o | q,k,v,o |
| Effective batch | 16 | 16 | 8 |
| Learning rate | 5e-5 | 2e-5 | 1e-5 |
| β (DPO) | 0.1 | — | — |
| KL coef (PPO) | — | — | 0.05 |
| Max length | 512 | 512 | 128 (prompt) + 64 (response) |
| Précision | bf16 LoRA | bf16 LoRA | bnb 4-bit base + bf16 LoRA |
| Epochs / Episodes | 1 epoch | 1 epoch | 1000 episodes |

## Pièges connus et workarounds

- **Windows + bnb optimizers** = segfault à l'import de TRL. Solution : Kaggle/Linux pour PPO.
- **Python 3.13** = pas de wheel torch CUDA. Solution : Python 3.11.
- **Kaggle T4×2** + `accelerate device_map="auto"` = sharding qui crashe PPOTrainer. Solution : `CUDA_VISIBLE_DEVICES=0` avant tout import.
- **PyArrow 19+** sur Linux = segfault à l'import de datasets. Solution : `pip install pyarrow==17.0.0`.
- **Windows cp1252** lors d'écriture JSON avec UTF-8 = `UnicodeEncodeError`. Solution : tous les `open()` ont `encoding="utf-8"` (déjà patché).
- **OneDrive sync** peut tronquer les fichiers entre Write Python et lecture bash. Solution : éviter de mettre le projet dans OneDrive si possible, sinon attendre la sync avant de lancer.

## Notes importantes

- **Jamais d'ETHICS pour le train.** Le sujet l'interdit, et notre split test reste vierge.
- **Le PPO utilise bnb 4-bit par défaut**. Pour bf16 pur, ajouter `--no_4bit` (nécessite ≥ 24 GB VRAM ou Qwen 0.5B).
- **Le PPO démarre depuis le DPO adapter** (warm-start). Si tu veux PPO depuis le baseline brut, modifie `train_ppo.py`.
- **Les checkpoints intermédiaires** (`results/*/checkpoint-*/`) sont gitignorés. Seul l'adapter final est versionné.
