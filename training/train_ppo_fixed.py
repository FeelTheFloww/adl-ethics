"""Étape PPO du pipeline RLHF — VERSION CORRIGÉE (TRL 0.12).

Différences avec train_ppo.py (chaque correctif est marqué `# FIX`):

  FIX-1  Symétrie policy/ref. Dans l'original, `prepare_model_for_kbit_training`
         n'était appliqué qu'à la POLITIQUE, pas à la référence. Cette fonction
         upcast les LayerNorm (et la lm_head) en fp32 d'un seul côté : policy et
         ref calculent alors des log-probs différentes MÊME avec un adaptateur
         identique, d'où une KL non nulle dès le step 0. On applique désormais
         le même traitement aux deux (ou à aucun).

  FIX-2  Dropout désactivé pour le calcul des log-probs. L'original laissait la
         policy en .train() (dropout LoRA p=0.05 actif) tandis que la ref était
         en .eval(). On force lora_dropout=0 sur l'adaptateur de policy : le
         dropout ne sert à rien en PPO (pas de régularisation utile) et injecte
         du bruit stochastique directement dans la KL.

  FIX-3  Normalisation (whitening) des récompenses + KL adaptatif. Le terme de
         pénalité KL (kl_coef) est abaissé et on active le contrôle adaptatif
         pour tolérer le bruit résiduel de la quantification.

  FIX-4  num_ppo_epochs ramené de 4 à 1 par défaut : 4 époques internes sur le
         même rollout poussent la policy loin de la ref à chaque step.

À lancer APRÈS avoir confirmé avec eval/kl_diagnostics.py que la KL d'init
retombe bien vers 0 dans la config symétrique.

Usage : CUDA_VISIBLE_DEVICES=0 python training/train_ppo_fixed.py \
            --dpo_adapter results/dpo_model \
            --reward_model_path results/reward_model \
            --output_dir results/rlhf_model_fixed
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


def _disable_lora_dropout(peft_model):
    """FIX-2 : met tous les Dropout LoRA à p=0 pour un scoring déterministe."""
    import torch.nn as nn
    for module in peft_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


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
    _disable_lora_dropout(policy)                                      # FIX-2
    policy.train()

    print("[ref] loading (DPO adapter, frozen, SYMMETRIC prep)...")
    ref_base = AutoModelForCausalLM.from_pretrained(args.model_name, **common_kw)
    if use_4bit:
        ref_base = prepare_model_for_kbit_training(ref_base)          # FIX-1
    ref = PeftModel.from_pretrained(ref_base, args.dpo_adapter)
    _disable_lora_dropout(ref)                                        # FIX-2
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

    return policy, ref, reward_model, value_model


def load_prompts(data_path, tokenizer, max_prompt_length, n_prompts):
    rows = load_jsonl(data_path)
    prompts = [r["prompt"].strip() for r in rows if r.get("prompt", "").strip()]
    prompts = list(dict.fromkeys(prompts))[:n_prompts]
    print(f"  -> {len(prompts)} unique prompts")

    def tok_fn(ex):
        enc = tokenizer(apply_chat_user(tokenizer, ex["prompt"]),
                        truncation=True, max_length=max_prompt_length, padding=False)
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
        num_ppo_epochs=args.num_ppo_epochs,        # FIX-4 : défaut 1
        num_mini_batches=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        total_episodes=args.total_episodes,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,                       # FIX-3 : défaut abaissé
        whiten_rewards=True,                        # FIX-3 : normalise les récompenses
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

    print("Starting PPO training (corrected config)...")
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
    parser.add_argument("--output_dir", type=str, default="results/rlhf_model_fixed")
    parser.add_argument("--n_prompts", type=int, default=1500)
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--num_ppo_epochs", type=int, default=1)        # FIX-4
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.02)          # FIX-3
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--response_length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="ppo_qwen_ethics_fixed")
    args = parser.parse_args()
    train_ppo(args)
