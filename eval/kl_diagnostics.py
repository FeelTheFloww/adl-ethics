"""Diagnostic de la divergence KL anormale du PPO (RLHF).

Objectif : déterminer POURQUOI la KL part déjà à ~110 au step 0 alors que la
politique et la référence sont warm-startées depuis le MÊME adaptateur DPO
(donc, en théorie, KL == 0 à l'initialisation).

Le script mesure la KL token-par-token entre politique et référence sur
quelques prompts, dans trois configurations, pour isoler la cause :

  (A) "buggy"  : reproduit train_ppo.py — `prepare_model_for_kbit_training`
                 est appliqué à la POLITIQUE mais PAS à la référence, et la
                 politique reste en mode train() (dropout LoRA actif).
  (B) "fixed"  : politique et référence traitées symétriquement, dropout
                 désactivé pour le scoring (model.eval()).
  (C) "bf16ref": référence chargée en bf16 (non quantifiée), politique en 4-bit.

Test de cohérence (sanity check) : KL(politique || politique) doit valoir 0.
S'il ne vaut pas 0, le bug est dans le calcul des log-probs, pas dans la
quantification.

Hypothèse testée : la KL ≈110 ne vient PAS d'une "accumulation de bruit de
quantification", mais d'une ASYMÉTRIE de traitement entre policy et ref
(prepare_model_for_kbit_training upcast les LayerNorm en fp32 d'un seul côté +
dropout actif d'un seul côté), qui crée un décalage de log-probs constant dès
le step 0.

Usage (Kaggle T4) :
    python eval/kl_diagnostics.py --dpo_adapter results/dpo_model \
        --data_path data/preferences.jsonl --n_prompts 16
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_adapter_model(model_name, adapter, *, four_bit, kbit_prepare, trainable):
    """Charge un modèle causal + adaptateur LoRA dans une configuration donnée."""
    kw = dict(trust_remote_code=True, device_map={"": 0})
    if four_bit:
        kw["quantization_config"] = bnb_config()
    else:
        kw["torch_dtype"] = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if four_bit and kbit_prepare:
        base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    return model


@torch.no_grad()
def seq_logprobs(model, input_ids, attention_mask, prompt_len, train_mode=False):
    """Somme des log-probs des tokens de RÉPONSE (au-delà de prompt_len).

    Renvoie aussi les log-probs par token pour inspection.
    train_mode=True garde le dropout actif (reproduit le bug policy.train()).
    """
    model.train(train_mode)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]            # prédit le token t+1
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # On ne garde que les positions de réponse (>= prompt_len-1 dans l'espace décalé)
    resp_mask = torch.zeros_like(tok_logp, dtype=torch.bool)
    resp_mask[:, prompt_len - 1:] = True
    resp_mask &= attention_mask[:, 1:].bool()
    return tok_logp, resp_mask


def kl_between(policy, ref, batch, prompt_len, *, policy_train):
    """KL(policy || ref) approximée par somme_token (logp_policy - logp_ref)."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    lp_pol, m = seq_logprobs(policy, ids, mask, prompt_len, train_mode=policy_train)
    lp_ref, _ = seq_logprobs(ref, ids, mask, prompt_len, train_mode=False)
    diff = (lp_pol - lp_ref) * m
    kl_per_seq = diff.sum(dim=1)                       # somme sur tokens (= métrique TRL)
    n_tok = m.sum(dim=1).clamp(min=1)
    kl_per_tok = kl_per_seq / n_tok
    return kl_per_seq.mean().item(), kl_per_tok.mean().item()


def build_batch(tokenizer, prompts, response_length, device):
    """Construit un batch (prompt formaté + réponse générée par un modèle neutre).

    Pour le diagnostic d'initialisation, le contenu exact de la réponse importe
    peu : on veut juste des tokens de réponse sur lesquels mesurer la KL.
    On concatène donc une réponse fixe courte.
    """
    texts, prompt_lens = [], []
    filler = " Yes, because it depends on the situation and the people involved."
    for p in prompts:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True,
        )
        plen = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
        texts.append(chat + filler)
        prompt_lens.append(plen)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=256, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    # prompt_len commun (le min) — suffisant pour le diagnostic
    return enc, min(prompt_lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=BASE_MODEL)
    ap.add_argument("--dpo_adapter", default="results/dpo_model")
    ap.add_argument("--data_path", default="data/preferences.jsonl")
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--response_length", type=int, default=64)
    ap.add_argument("--out", default="results/kl_diagnostics.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prompts
    prompts = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = (r.get("prompt") or "").strip()
            if p:
                prompts.append(p)
            if len(prompts) >= args.n_prompts:
                break
    print(f"{len(prompts)} prompts chargés.")

    batch, prompt_len = build_batch(tokenizer, prompts, args.response_length, device)
    results = {}

    # ---- (A) Reproduction du bug : policy avec kbit_prepare + train(), ref sans ----
    print("\n[A] Config 'buggy' (reproduit train_ppo.py)")
    pol = load_adapter_model(args.model_name, args.dpo_adapter,
                             four_bit=True, kbit_prepare=True, trainable=True)
    ref = load_adapter_model(args.model_name, args.dpo_adapter,
                             four_bit=True, kbit_prepare=False, trainable=False)
    kseq, ktok = kl_between(pol, ref, batch, prompt_len, policy_train=True)
    self_seq, _ = kl_between(pol, pol, batch, prompt_len, policy_train=False)
    print(f"  KL(policy||ref) = {kseq:.2f} (somme) | {ktok:.4f} (par token)")
    print(f"  sanity KL(policy||policy, eval) = {self_seq:.4f}  (doit valoir ~0)")
    results["A_buggy"] = {"kl_sum": kseq, "kl_per_token": ktok, "kl_self": self_seq}
    del pol, ref
    torch.cuda.empty_cache()

    # ---- (B) Symétrique, dropout off des deux côtés ----
    print("\n[B] Config 'fixed' (symétrique, eval/eval)")
    pol = load_adapter_model(args.model_name, args.dpo_adapter,
                             four_bit=True, kbit_prepare=True, trainable=True)
    ref = load_adapter_model(args.model_name, args.dpo_adapter,
                             four_bit=True, kbit_prepare=True, trainable=False)
    kseq, ktok = kl_between(pol, ref, batch, prompt_len, policy_train=False)
    print(f"  KL(policy||ref) = {kseq:.2f} (somme) | {ktok:.4f} (par token)")
    results["B_fixed_symmetric"] = {"kl_sum": kseq, "kl_per_token": ktok}
    del pol, ref
    torch.cuda.empty_cache()

    # ---- (C) Référence en bf16 (non quantifiée) ----
    print("\n[C] Config 'bf16ref' (policy 4-bit, ref bf16)")
    pol = load_adapter_model(args.model_name, args.dpo_adapter,
                             four_bit=True, kbit_prepare=True, trainable=True)
    ref = load_adapter_model(args.model_name, args.dpo_adapter,
                             four_bit=False, kbit_prepare=False, trainable=False)
    kseq, ktok = kl_between(pol, ref, batch, prompt_len, policy_train=False)
    print(f"  KL(policy||ref) = {kseq:.2f} (somme) | {ktok:.4f} (par token)")
    results["C_bf16_ref"] = {"kl_sum": kseq, "kl_per_token": ktok}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats -> {args.out}")
    print("\nLecture : si A >> B, la KL anormale vient de l'asymétrie de "
          "traitement (kbit_prepare + dropout), PAS du bruit de quantification.")


if __name__ == "__main__":
    main()
