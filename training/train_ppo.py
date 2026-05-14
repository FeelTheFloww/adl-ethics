"""
training/train_ppo.py
─────────────────────
RLHF — étape 2 : optimisation de politique par PPO en utilisant le reward model
entraîné précédemment.

ATTENTION VRAM : PPO charge en mémoire 4 modèles
  - policy (avec LoRA)    : ~1.5 GB en 4-bit
  - ref model (frozen)    : peut être désactivé en utilisant le PEFT adapter switch
  - value head            : petit
  - reward model (frozen) : ~1.5 GB en 4-bit
Sur RTX 4060 8 GB, c'est tendu. On utilise un batch_size=1 + grad_accum élevé
et `gradient_checkpointing`. Si OOM, deux options :
  1. Réduire max_length à 128
  2. Déplacer sur Kaggle T4×2

API TRL : on utilise PPOTrainer (TRL ≥ 0.12).

Usage :
  python training/train_ppo.py --reward_model_path results/reward_model
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


# ── Config ────────────────────────────────────────────────────────────────────
def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ── Chargement des modèles ────────────────────────────────────────────────────
def load_policy(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base = get_peft_model(base, get_lora_config())

    # Ajoute la value head pour PPO
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base)
    return policy, tokenizer


def load_reward_model(reward_model_path: str, base_model_name: str):
    """Charge le RM (base + adapter LoRA) en 4-bit, mode eval."""
    from transformers import AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if base.config.pad_token_id is None:
        base.config.pad_token_id = tokenizer.pad_token_id
    rm = PeftModel.from_pretrained(base, reward_model_path)
    rm.eval()
    return rm, tokenizer


# ── Données : prompts seuls (pas de chosen/rejected en PPO) ───────────────────
def load_prompts(data_path: str, tokenizer, max_prompt_length: int):
    print(f"Loading prompts from {data_path}…")
    prompts = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = ex.get("prompt", "").strip()
            if p:
                prompts.append(p)
    # Dedupe
    prompts = list(dict.fromkeys(prompts))
    print(f"  -> {len(prompts)} unique prompts")

    def tok(p):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(formatted, truncation=True, max_length=max_prompt_length)
        return {"input_ids": enc["input_ids"], "query": formatted}

    ds = Dataset.from_dict({"prompt": prompts}).map(lambda x: tok(x["prompt"]))
    return ds


# ── Reward scoring ────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_rewards(rm, rm_tokenizer, queries: list[str], responses: list[str], device) -> list[torch.Tensor]:
    """Pour chaque (query, response), score scalaire renvoyé par le RM."""
    rewards = []
    for q, r in zip(queries, responses):
        text = q + r  # query inclut déjà le template; r est la suite générée
        enc = rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        out = rm(**enc)
        score = out.logits[0, 0]  # scalar reward
        rewards.append(score.detach())
    return rewards


# ── Training ──────────────────────────────────────────────────────────────────
def train_ppo(args):
    policy, tokenizer = load_policy(args.model_name)
    rm, rm_tokenizer = load_reward_model(args.reward_model_path, args.model_name)
    device = next(policy.parameters()).device

    train_ds = load_prompts(args.data_path, tokenizer, args.max_prompt_length)
    # PPO consomme typiquement quelques centaines/milliers de prompts par epoch
    train_ds = train_ds.select(range(min(args.n_prompts, len(train_ds))))

    ppo_config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        kl_coef=args.kl_coef,
        cliprange=0.2,
        cliprange_value=0.2,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        log_with="wandb" if args.wandb else None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        tokenizer=tokenizer,
        dataset=train_ds,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }

    print("Starting PPO training…")
    step = 0
    for epoch in range(args.epochs):
        for batch in trainer.dataloader:
            queries = batch["query"]
            query_tensors = [torch.tensor(ids).to(device) for ids in batch["input_ids"]]
            # Génération
            response_tensors = trainer.generate(query_tensors, return_prompt=False, **gen_kwargs)
            responses_text = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            # Reward
            rewards = compute_rewards(rm, rm_tokenizer, queries, responses_text, device)
            # PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)
            trainer.log_stats(stats, batch, rewards)
            step += 1
            if step % 10 == 0:
                mean_r = sum(float(r) for r in rewards) / max(1, len(rewards))
                print(f"  step {step} | mean reward = {mean_r:.3f}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    # Sauvegarde de l'adapter de la policy
    trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reward_model_path", type=str, default="results/reward_model")
    parser.add_argument("--data_path", type=str, default="data/preferences.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/rlhf_model")
    parser.add_argument("--n_prompts", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Hard cap on PPO steps (safety on 8GB).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="ppo_qwen_ethics")
    args = parser.parse_args()
    train_ppo(args)
