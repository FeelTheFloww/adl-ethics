"""Génère des paires de préférences (chosen/rejected) au format proche d'ETHICS via
un LLM externe. Aucun exemple d'ETHICS n'est utilisé en entrée.

Providers : anthropic / openai / groq — clé API via ANTHROPIC_API_KEY,
OPENAI_API_KEY ou GROQ_API_KEY.

Usage : python data/generate_synthetic.py --provider anthropic --n 3000
"""

import argparse
import json
import os
import random
import time
from typing import Optional


SEED_TOPICS = [
    "everyday social interactions",
    "honesty and lying",
    "fairness in resource sharing",
    "respect for autonomy",
    "harm avoidance",
    "trade-offs between competing duties",
    "promise-keeping",
    "loyalty vs. integrity",
    "consent in interpersonal actions",
    "compassion vs. justice",
    "small dishonesties for kindness",
    "rule-following vs. exceptions",
    "personal responsibility for outcomes",
    "stewardship of resources",
    "small moral compromises",
]

FRAMEWORKS = [
    "consequentialist / utilitarian reasoning",
    "deontological reasoning (duty, rules)",
    "virtue ethics (character)",
    "common-sense morality",
    "justice as fairness",
]


SYSTEM_PROMPT = """You are a careful ethics tutor producing training data for an alignment dataset.
Your output must be a strict JSON object — no commentary, no markdown fences."""

USER_PROMPT_TEMPLATE = """Produce ONE training example for ethical alignment.

The example must have:
- "prompt": a short, concrete first-person or third-person ethical SCENARIO (1-3 sentences),
            followed by a yes/no moral question (e.g., "Is this action morally wrong?",
            "Was this excuse reasonable?", "Did the character exemplify {framework}?").
- "chosen": a 1-2 sentence answer that starts with a clear yes/no, then explains using
            {framework}. The answer must be morally correct and well-grounded.
- "rejected": a 1-2 sentence answer that arrives at the OPPOSITE conclusion, or one that
              is superficial / inconsistent / morally flawed.

Topic seed: {topic}

Output exactly this JSON shape (no other text):
{{"prompt": "...", "chosen": "...", "rejected": "..."}}"""


# Providers
def call_anthropic(system: str, user: str, model: str = "claude-haiku-4-5-20251001") -> Optional[str]:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_openai(system: str, user: str, model: str = "gpt-4o-mini") -> Optional[str]:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content


def call_groq(system: str, user: str, model: str = "llama-3.3-70b-versatile") -> Optional[str]:
    """Groq free tier (Llama-3.3-70B), API compatible OpenAI."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=512,
        temperature=0.8,
    )
    return resp.choices[0].message.content


def call_llm(provider: str, system: str, user: str) -> Optional[str]:
    if provider == "anthropic":
        return call_anthropic(system, user)
    if provider == "openai":
        return call_openai(system, user)
    if provider == "groq":
        return call_groq(system, user)
    raise ValueError(f"Unknown provider: {provider}")


# Parsing
def parse_pair(raw: str) -> Optional[dict]:
    """Extrait l'objet JSON de la sortie du LLM."""
    raw = (raw or "").strip()
    if raw.startswith("```"):  # retire les fences markdown
        raw = raw.strip("`")
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:  # repli : extrait le premier objet JSON
            start = raw.index("{")
            end = raw.rindex("}") + 1
            obj = json.loads(raw[start:end])
        except Exception:
            return None
    if not all(k in obj for k in ("prompt", "chosen", "rejected")):
        return None
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in ("prompt", "chosen", "rejected")):
        return None
    return {"prompt": obj["prompt"].strip(),
            "chosen": obj["chosen"].strip(),
            "rejected": obj["rejected"].strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq"], default="groq")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out_path", type=str, default="data/synthetic_ethics.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between calls (rate limit).")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    # Mode append : permet de reprendre une génération interrompue
    existing = 0
    if os.path.exists(args.out_path):
        with open(args.out_path, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        print(f"[Resume] {existing} examples already present.")

    n_to_make = max(0, args.n - existing)
    print(f"Generating {n_to_make} new examples via {args.provider}…")

    written = 0
    with open(args.out_path, "a", encoding="utf-8") as f:
        for i in range(n_to_make):
            topic = random.choice(SEED_TOPICS)
            framework = random.choice(FRAMEWORKS)
            user_prompt = USER_PROMPT_TEMPLATE.format(topic=topic, framework=framework)
            try:
                raw = call_llm(args.provider, SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                print(f"  [{i}] API error: {e}")
                time.sleep(2.0)
                continue
            obj = parse_pair(raw)
            if obj is None:
                continue
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if (i + 1) % 50 == 0:
                print(f"  …{i+1}/{n_to_make} ({written} valid)")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote {written} new examples (total: {existing + written}) to {args.out_path}")


if __name__ == "__main__":
    main()
