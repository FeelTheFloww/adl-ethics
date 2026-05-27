"""Étape Reward Model du pipeline RLHF.

Pipeline : (1) data  →  (2) DPO  →  [3] reward model  →  (4) PPO  →  (5) eval
Entrée   : data/preferences.jsonl
Sortie   : results/reward_model/  (adaptateur LoRA + tête `score`)

La tête `score` (num_labels=1) est initialisée aléatoirement et apprise via
LoRA grâce à modules_to_save=["score"]. Le chat template est appliqué avant
tokenisation pour que le RM voie les séquences au même format que le DPO/PPO.

Usage : python training/train_reward_model.py --data_path data/preferences.jsonl
"""

import argparse

import torch
from datasets import Dataset
from peft import TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from _common import (
    BASE_MODEL, apply_chat_pair, load_jsonl, load_tokenizer, make_lora_config,
)


def load_reward_model(model_name: str, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(
        model,
        make_lora_config(TaskType.SEQ_CLS, modules_to_save=["score"]),
    )
    model.print_trainable_parameters()
    return model


def load_reward_dataset(data_path: str, tokenizer, max_length: int,
                        eval_ratio: float = 0.05):
    print(f"Loading preferences from {data_path}…")
    rows = load_jsonl(data_path)
    print(f"  -> {len(rows)} raw pairs")

    def tok(example):
        chosen_text = apply_chat_pair(tokenizer, example["prompt"], example["chosen"])
        rejected_text = apply_chat_pair(tokenizer, example["prompt"], example["rejected"])
        tc = tokenizer(chosen_text, truncation=True, max_length=max_length)
        tr = tokenizer(rejected_text, truncation=True, max_length=max_length)
        return {
            "input_ids_chosen": tc["input_ids"],
            "attention_mask_chosen": tc["attention_mask"],
            "input_ids_rejected": tr["input_ids"],
            "attention_mask_rejected": tr["attention_mask"],
        }

    ds = Dataset.from_list(rows)
    ds = ds.map(tok, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=eval_ratio, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")
    return split["train"], split["test"]


def train_reward_model(args):
    tokenizer = load_tokenizer(args.model_name)
    model = load_reward_model(args.model_name, tokenizer)
    train_ds, eval_ds = load_reward_dataset(args.data_path, tokenizer, args.max_length)

    training_args = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=False,
        max_length=args.max_length,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting reward model training…")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=BASE_MODEL)
    parser.add_argument("--data_path", type=str, default="data/preferences.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/reward_model")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="reward_model_qwen_ethics")
    args = parser.parse_args()
    train_reward_model(args)
