#!/bin/bash
# ============================================================
# ADL — Pipeline complète d'alignement éthique
# ============================================================
# Exécute les 5 étapes du pipeline dans l'ordre, en local (Linux/WSL/Kaggle).
# Sur Kaggle, préférer les notebooks dédiés (notebooks/01-04) pour gérer la
# limite des 12 h par session et l'export des adaptateurs entre étapes.
#
# Usage : bash run_pipeline.sh
# Prérequis : avoir lancé setup.sh une fois.

set -e

export CUDA_VISIBLE_DEVICES=0   # obligatoire pour PPO sur multi-GPU (Kaggle T4×2)

DATA_PATH="data/preferences.jsonl"
CORPUS="data/ethical_corpus_v2.json"
DPO_DIR="results/dpo_model"
RM_DIR="results/reward_model"
RLHF_DIR="results/rlhf_model"
EVAL_OUT="results/eval_results.json"

echo "============================================================"
echo " [1/5] Préparation des données (PKU + UltraFeedback)"
echo "============================================================"
python data/prepare_preferences.py \
    --n_pku 15000 --n_ultra 5000 \
    --out_path "$DATA_PATH"

echo ""
echo "============================================================"
echo " [2/5] Entraînement DPO"
echo "============================================================"
python training/train_dpo.py \
    --data_path "$DATA_PATH" --output_dir "$DPO_DIR" \
    --batch_size 1 --grad_accum 16 --max_length 384

echo ""
echo "============================================================"
echo " [3/5] Entraînement Reward Model (étape 1 RLHF)"
echo "============================================================"
python training/train_reward_model.py \
    --data_path "$DATA_PATH" --output_dir "$RM_DIR" \
    --batch_size 1 --grad_accum 16 --max_length 384

echo ""
echo "============================================================"
echo " [4/5] Entraînement PPO (étape 2 RLHF)"
echo "============================================================"
python training/train_ppo.py \
    --dpo_adapter "$DPO_DIR" --reward_model_path "$RM_DIR" \
    --data_path "$DATA_PATH" --output_dir "$RLHF_DIR"

echo ""
echo "============================================================"
echo " [5/5] Évaluation sur ETHICS (baseline / DPO / RLHF / RAG)"
echo "============================================================"
python eval/evaluate_ethics.py \
    --dpo_adapter "$DPO_DIR" \
    --rlhf_adapter "$RLHF_DIR" \
    --ethical_corpus "$CORPUS" \
    --output_path "$EVAL_OUT" \
    --n_per_cat 100

echo ""
echo "============================================================"
echo " [bonus] Figure KL pour le rapport"
echo "============================================================"
python eval/plot_ppo_kl.py \
    --trainer_state "$RLHF_DIR/checkpoint-125/trainer_state.json" \
    --out results/ppo_kl_divergence.png

echo ""
echo "Pipeline terminée. Résultats : $EVAL_OUT"
