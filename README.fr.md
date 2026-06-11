# ADL — Alignement Éthique de Petits Modèles de Langage

[English](README.md) | **Français**

Comparaison de trois stratégies d'alignement (DPO, RLHF via PPO, RAG) sur le benchmark ETHICS (Hendrycks et al., 2021) avec **Qwen2.5-1.5B-Instruct** comme modèle de base.

> **Résultat principal** : aucune méthode n'améliore significativement le modèle de base (64,6 % de précision globale). L'instruction tuning préalable du modèle crée un effet de plafond qui limite toute marge de progression, indépendamment de la méthode d'alignement utilisée. Le pipeline PPO diverge sous quantification 4-bit (KL ~110, cible < 0,5). La RAG montre le signal le plus clair sur l'utilitarisme (+6 pts) mais régresse sur la vertu (−4 pts).

---

## Structure du projet

```
ADL/
├── data/
│   ├── prepare_preferences.py     # Construit le jeu hybride PKU + UltraFeedback (+ synthétique)
│   ├── generate_synthetic.py      # Génère ~3 000 paires synthétiques via Anthropic/OpenAI/Groq
│   ├── ethical_corpus.json        # Corpus RAG v1 (21 documents)
│   ├── ethical_corpus_v2.json     # Corpus RAG v2 (170 documents — recommandé)
│   ├── preferences.jsonl          # Données d'entraînement finales (format JSONL)
│   └── synthetic_ethics.jsonl     # Paires synthétiques générées (optionnel)
├── training/
│   ├── train_dpo.py               # DPO avec LoRA bf16 (TRL DPOTrainer)
│   ├── train_reward_model.py      # Reward model — étape 1 RLHF (TRL RewardTrainer)
│   └── train_ppo.py               # PPO avec bnb 4-bit — étape 2 RLHF (TRL PPOTrainer)
├── eval/
│   ├── evaluate_ethics.py         # Évaluation logit-based + RAG intégré (script canonique)
│   └── plot_ppo_kl.py             # Trace la courbe KL du PPO (figure du rapport)
├── notebooks/                     # Notebooks Kaggle prêts à l'emploi (un par phase)
│   ├── 01_dpo.ipynb               # Phase 1 : train DPO adapter
│   ├── 02_reward_model.ipynb      # Phase 2 : train Reward Model
│   ├── 03_ppo.ipynb               # Phase 3 : train PPO (attache DPO + RM)
│   └── 04_eval.ipynb              # Phase 4 : eval finale sur ETHICS
├── results/                       # Adapters LoRA entraînés, scores JSON, graphes
├── report/
│   ├── main.tex                   # Rapport format ACL
│   └── references.bib
└── setup.sh                       # Script d'installation Linux/Kaggle
```

Le corpus de principes éthiques (`data/ethical_corpus_v2.json`, 170 documents) est chargé directement par `eval/evaluate_ethics.py` : l'index FAISS est construit en mémoire (embedder `sentence-transformers/all-MiniLM-L6-v2`) à chaque évaluation. Pas d'étape de build d'index séparée.

---

## Environnement

| Plateforme | Statut | Notes |
|---|---|---|
| Kaggle T4 (16 GB) | ✅ Recommandé | Fixer `CUDA_VISIBLE_DEVICES=0` avant tout import |
| Linux + GPU ≥ 8 GB | ✅ OK | bnb 4-bit + bf16, Python 3.11 |
| Windows natif | ⚠️ Partiel | DPO + RM OK, PPO crashe (segfault bnb optimizers) |

**Python recommandé : 3.11** (3.12 OK — 3.13 incompatible avec les wheels torch CUDA et bnb).

### Installation

```bash
pip install -q -U bitsandbytes transformers==4.46.3 trl==0.12.0 peft==0.14.0 \
    accelerate==1.2.0 datasets==3.2.0 sentence-transformers faiss-cpu \
    anthropic openai pyarrow==17.0.0 tqdm
```

Sur Kaggle : **Restart Session** après installation (obligatoire pour le rechargement des libs CUDA).

---

## Données d'entraînement

| Source | Taille | Rôle |
|---|---|---|
| PKU-SafeRLHF | ~15 000 paires | Préférences humaines safety/harmlessness |
| UltraFeedback | ~5 000 paires | Préférences générales haute qualité |
| Synthétique (optionnel) | ~3 000 paires | Scénarios moraux générés via Anthropic Claude Haiku |

Format final : JSONL avec champs `prompt`, `chosen`, `rejected`, `source`.

Les données synthétiques ont été intégrées dans une variante d'entraînement mais n'ont produit aucune amélioration mesurable sur ETHICS — l'effet de plafond du modèle de base domine.

**Contrainte stricte** : aucun exemple du benchmark ETHICS n'est utilisé à l'entraînement.

---

## Workflow Kaggle multi-datasets (recommandé)

À cause de la limite de **12 h par session Kaggle**, on splitte le pipeline en **4 notebooks indépendants** (un par phase), avec **export d'un Kaggle Dataset entre chaque étape** pour transporter les adapters d'un run au suivant.

Les 4 notebooks sont prêts à l'emploi dans `notebooks/` :

| Notebook | Datasets à attacher (Add Data) | Produit | Durée approx. |
|---|---|---|---|
| `01_dpo.ipynb` | aucun | DPO adapter → **`adl-dpo-adapter`** | ~10 h |
| `02_reward_model.ipynb` | aucun | RM adapter → **`adl-reward-model`** | ~8 h |
| `03_ppo.ipynb` | **`adl-dpo-adapter` ET `adl-reward-model`** | RLHF adapter → **`adl-rlhf-model`** | ~5 h |
| `04_eval.ipynb` | `adl-dpo-adapter` + `adl-rlhf-model` | `eval_results.json` | ~30 min |

**Pattern entre deux notebooks** :
1. À la fin du notebook N, `shutil.make_archive` zippe `results/<phase>_model/` → `/kaggle/working/<phase>_model.zip`
2. Save & Run All → Output → **New Dataset** depuis ce zip
3. Au début du notebook N+1, **Add Data** → attache le ou les datasets nécessaires
4. Première cellule décompresse `/kaggle/input/<dataset-name>/*.zip` vers `results/<phase>_model/`

**Important pour le PPO** : le notebook 03 a besoin des **deux** datasets (DPO + RM) attachés en même temps, car le PPO warm-starte sa policy depuis le DPO et utilise le RM comme signal de récompense.

---

## Pipeline local équivalent (Linux/WSL)

Si tu as une machine Linux avec GPU ≥ 16 GB et que tu veux tout faire dans une seule session :

```bash
# 1. Données
python data/prepare_preferences.py --n_pku 15000 --n_ultra 5000 --out_path data/preferences.jsonl

# 2. DPO
python training/train_dpo.py --data_path data/preferences.jsonl \
    --output_dir results/dpo_model --batch_size 1 --grad_accum 16 --max_length 384

# 3. Reward Model
python training/train_reward_model.py --data_path data/preferences.jsonl \
    --output_dir results/reward_model --batch_size 1 --grad_accum 16 --max_length 384

# 4. PPO
python training/train_ppo.py --dpo_adapter results/dpo_model \
    --reward_model_path results/reward_model --data_path data/preferences.jsonl \
    --output_dir results/rlhf_model

# 5. Eval finale
python eval/evaluate_ethics.py --dpo_adapter results/dpo_model \
    --rlhf_adapter results/rlhf_model --ethical_corpus data/ethical_corpus_v2.json \
    --output_path results/eval_results.json --n_per_cat 100

# 6. (Optionnel) Figure KL du PPO
python eval/plot_ppo_kl.py \
    --trainer_state results/rlhf_model/checkpoint-125/trainer_state.json \
    --out results/ppo_kl_divergence.png
```

---

## Hyperparamètres

| Paramètre | DPO | Reward Model | PPO |
|---|---|---|---|
| LoRA rank / alpha | 16 / 32 | 16 / 32 | 16 / 32 |
| Target modules | q,k,v,o | q,k,v,o | q,k,v,o |
| Dropout | 0.05 | 0.05 | 0.05 |
| Batch effectif | 16 (1×16) | 16 (1×16) | 8 (2×4) |
| Learning rate | 5e-5 | 2e-5 | 1e-5 |
| β (DPO) / KL coef (PPO) | 0.1 | — | 0.05 |
| Max length (prompt + réponse) | 384 / 256 | 384 | 128 / 64 |
| Précision | bf16 LoRA | bf16 LoRA | bnb 4-bit base + bf16 LoRA |
| Durée | 1 epoch (1 029 steps) | 1 epoch | 1 000 épisodes (125 steps gradient) |
| Warmup | 10 % | — | — |
| Eval / Save | steps 200 / 100 | steps 200 / 100 | steps — / 200 |

### PPO : 4 modèles simultanés en mémoire

Le PPO charge **4 instances** du backbone Qwen2.5-1.5B sur le même GPU :
1. **Policy** (warm-startée depuis le DPO adapter, entraînable)
2. **Référence** (DPO adapter gelé — contrainte KL)
3. **Reward model** (gelé)
4. **Value model** (warm-starté depuis le RM, entraînable)

Sous bnb 4-bit, le bruit numérique cumulé entre ces 4 modèles provoque une divergence KL explosive (~110 vs cible < 0,5), rendant l'optimisation ineffective.

---

## Évaluation

Le script `eval/evaluate_ethics.py` utilise une méthode **logit-based** sans génération :

- Pour chaque exemple, on compare `P(token "0")` vs `P(token "1")` sur le dernier token du prompt (ou `P("A")` vs `P("B")` pour l'utilitarisme).
- Le prompt passe par le chat template Qwen (`apply_chat_template`) avant tokenisation.
- Rapide et déterministe : pas de sampling, pas de variabilité.
- 5 sous-ensembles × 100 exemples = **500 exemples au total**.
- Pour la condition RAG : on récupère `k=3` principes éthiques (similarité cosinus avec MiniLM, index FAISS construit en RAM) et on les injecte en tête du prompt.

Conditions évaluées : `baseline`, `dpo`, `rlhf`, `rag`.

---

## Pièges connus

| Problème | Cause | Solution |
|---|---|---|
| PPO crashe / KL explose | 4 modèles en 4-bit sur 16 GB | Kaggle T4, `CUDA_VISIBLE_DEVICES=0` avant tout import |
| Segfault PPO sous Windows | bnb optimizers incompatibles Windows | Utiliser Kaggle ou Linux |
| `datasets` segfault au chargement | PyArrow 19+ | `pip install pyarrow==17.0.0` |
| Pas de wheel torch CUDA | Python 3.13 | Utiliser Python 3.11 |
| Sharding PPO sur Kaggle T4×2 | `accelerate` détecte 2 GPU | `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` |
| `UnicodeEncodeError` JSON | Windows cp1252 | Tous les `open()` ont `encoding="utf-8"` |
| OOM DPO/RM batch=4 sur T4 | bf16 sans bnb sur Qwen 1.5B | `--batch_size 1 --grad_accum 16 --max_length 384` |
