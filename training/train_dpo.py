"""
training/train_dpo.py
─────────────────────
DPO QLoRA sur Qwen2.5-1.5B avec les données de préférence hybrides.

Usage :
  python training/train_dpo.py --data_path data/preferences.jsonl
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_for_dpo(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()
    return model, tokenizer


def format_with_chat_template(example, tokenizer):
    prompt_text = example["prompt"]
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": formatted_prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def load_preferences(data_path, tokenizer, eval_ratio=0.05):
    print(f"Loading preferences from {data_path}...")
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  -> {len(rows)} raw pairs")
    ds = Dataset.from_list(rows)
    ds = ds.map(
        lambda ex: format_with_chat_template(ex, tokenizer),
        remove_columns=[c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")],
    )
    split = ds.train_test_split(test_size=eval_ratio, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")
    return split["train"], split["test"]


def train_dpo(args):
    model, tokenizer = load_model_for_dpo(args.model_name)
    train_ds, eval_ds = load_preferences(args.data_path, tokenizer)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=False,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/preferences.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/dpo_model")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="dpo_qwen_ethics")
    args = parser.parse_args()
    train_dpo(args)
