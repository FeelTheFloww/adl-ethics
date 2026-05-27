"""Helpers partagés par les trois scripts d'entraînement (DPO, RM, PPO).

Centralise la configuration LoRA, la quantification 4-bit, le chargement du
tokenizer et la lecture du JSONL de préférences pour éviter la duplication.
Chaque fichier d'entraînement reste l'entrée unique pour son type
(train_dpo.py, train_reward_model.py, train_ppo.py).
"""

import json
from typing import Optional

import torch
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig


# Modèle de base et hyperparamètres LoRA partagés (cf. tableau du rapport)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def make_lora_config(task_type: TaskType,
                     modules_to_save: Optional[list[str]] = None) -> LoraConfig:
    """Crée la LoraConfig commune (r=16, alpha=32, dropout=0,05, projections d'attention).

    task_type : TaskType.CAUSAL_LM pour DPO/PPO, TaskType.SEQ_CLS pour le reward model.
    modules_to_save : modules entièrement entraînables (ex. ["score"] pour la tête RM).
    """
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=task_type,
        modules_to_save=modules_to_save,
    )


def get_bnb_config() -> BitsAndBytesConfig:
    """Configuration QLoRA 4-bit (NF4 + double quant, compute bf16). Utilisée par le PPO."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_name: str = BASE_MODEL):
    """Charge le tokenizer Qwen et fixe pad_token = eos_token si absent."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_jsonl(path: str) -> list[dict]:
    """Lit un fichier JSONL en ignorant silencieusement les lignes invalides."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def apply_chat_user(tokenizer, text: str) -> str:
    """Formate un message utilisateur via le chat template (DPO / PPO)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def apply_chat_pair(tokenizer, prompt: str, response: str) -> str:
    """Formate une paire (user, assistant) via le chat template (reward model)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False,
    )
