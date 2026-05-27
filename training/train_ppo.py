"""Étape PPO du pipeline RLHF (TRL 0.12).

Pipeline : (1) data  →  (2) DPO  →  (3) reward model  →  [4] PPO  →  (5) eval
Entrées  : results/dpo_model/   (warm-start de la policy et de la référence)
           results/reward_model/ (reward + value model warm-startés)
Sortie   : results/rlhf_model/   (adaptateur LoRA RLHF)

Conçu pour Kaggle T4 16 GB en bnb 4-bit. Charge simultanément quatre modèles
(policy, ref, reward, value) dans le même GPU.

IMPORTANT : sur Kaggle T4×2, exporter CUDA_VISIBLE_DEVICES=0 AVANT tout import
torch — sinon accelerate distribue les modèles sur les deux GPU et casse
PPOTrainer.

Usage : python training/train_ppo.py --dpo_adapter results/dpo_model \\
            --reward_model_path results/reward_model \\
            --output_dir results/rlhf_model
"""

import argparse
import sys

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer

from _common import (
    BASE_MODEL, apply_chat_user, get_bnb_config, load_jsonl, load_tokenizer,
)


def load_models(args, tokenizer):
    use_4bit = not args.no_4bit
    common_kw = dict(trust_remote_code=True, device_map={"": 0})
    if use_4bit:
        common_kw["quantization_config"] = get_bnb_config()
    else:
        common_kw["torch_dtype"] = torch.bfloat16

    print("[policy] loading (warm-started from DPO, trainable)...")
    policy_base = AutoModelForCausalLM.from_pretrained(args.model_name, **common_kw)
    if use_4bit:
        policy_base = prepare_model_for_kbit_training(policy_base)
    policy = PeftModel.from_pretrained(policy_base, args.dpo_adapter, is_trainable=True)
    policy.train()

    print("[ref] loading (DPO adapter, frozen)...")
    ref_base = AutoModelForCausalLM.from_pretrained(args.model_name, **common_kw)
    ref = PeftModel.from_pretrained(ref_base, args.dpo_adapter)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    print("[reward] loading (RM, frozen)...")
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1, **common_kw)
    if rm_base.config.pad_token_id is None:
        rm_base.config.pad_token_id = tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(rm_base, args.reward_model_path)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    print("[value] loading (warm-started from RM, trainable)...")
    value_base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1, **common_kw)
    if value_base.config.pad_token_id is None:
        value_base.config.pad_token_id = tokenizer.pad_token_id
    if use_4bit:
        value_base = prepare_model_for_kbit_training(value_base)
    value_model = PeftModel.from_pretrained(value_base, args.reward_model_path, is_trainable=True)
    value_model.train()

    for name, m in [("policy", policy), ("ref", ref),
                    ("reward", reward_model), ("value", value_model)]:
        devs = set(str(p.device) for p in m.parameters())
        print(f"  {name}: devices={devs}")
        if len(devs) > 1:
            print(f"  WARNING: {name} is sharded. "
                  "Set CUDA_VISIBLE_DEVICES=0 before launching.", file=sys.stderr)

    return policy, ref, reward_model, value_model


def load_prompts(data_path: str, tokenizer, max_prompt_length: int, n_prompts: int):
    print(f"Loading prompts from {data_path}...")
    rows = load_jsonl(data_path)
    prompts = [r["prompt"].strip() for r in rows if r.get("prompt", "").strip()]
    prompts = list(dict.fromkeys(prompts))[:n_prompts]
    print(f"  -> {len(prompts)} unique prompts")

    def tok_fn(ex):
        enc = tokenizer(
            apply_chat_user(tokenizer, ex["prompt"]),
            truncation=True, max_length=max_prompt_length, padding=False,
        )
        return {"input_ids": enc["input_ids"]}

    return Dataset.from_dict({"prompt": prompts}).map(tok_fn, remove_columns=["prompt"])


def train_ppo(args):
    if torch.cuda.device_count() > 1:
        print(f"WARNING: {torch.cuda.device_count()} GPUs visible. "
              "Set CUDA_VISIBLE_DEVICES=0 BEFORE importing torch.", file=sys.stderr)

    tokenizer = load_tokenizer(args.model_name)
    policy, ref, reward_model, value_model = load_models(args, tokenizer)
    train_ds = load_prompts(args.data_path, tokenizer, args.max_prompt_length, args.n_prompts)

    config = PPOConfig(
        output_dir=args.output_dir,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        total_episodes=args.total_episodes,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        remove_unused_columns=False,
        response_length=args.response_length,
        temperature=args.temperature,
        num_sample_generations=0,
    )

    trainer = PPOTrainer(
        config=config,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_ds,
    )

    print("Starting PPO training (TRL 0.12 API)...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=BASE_MODEL)
    parser.add_argument("--dpo_adapter", type=str, default="results/dpo_model")
    parser.add_argument("--reward_model_path", type=str, default="results/reward_model")
    parser.add_argument("--data_path", type=str, default="data/preferences.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/rlhf_model")
    parser.add_argument("--n_prompts", type=int, default=1500)
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--num_ppo_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--response_length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="ppo_qwen_ethics")
    args = parser.parse_args()
    train_ppo(args)
