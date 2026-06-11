# ADL — Ethical Alignment of Small Language Models

**English** | [Français](README.fr.md)

Comparison of three alignment strategies (DPO, RLHF via PPO, RAG) on the ETHICS benchmark (Hendrycks et al., 2021), using **Qwen2.5-1.5B-Instruct** as the base model.

> **Key finding**: none of the methods significantly improves on the base model (64.6% overall accuracy). The model's prior instruction tuning creates a ceiling effect that limits any room for improvement, regardless of the alignment method. The PPO pipeline diverges under 4-bit quantization (KL ~110 vs. target < 0.5). RAG shows the clearest signal on utilitarianism (+6 pts) but regresses on virtue (−4 pts).

---

## Project structure

```
ADL/
├── data/
│   ├── prepare_preferences.py     # Builds the hybrid PKU + UltraFeedback (+ synthetic) dataset
│   ├── generate_synthetic.py      # Generates ~3,000 synthetic pairs via Anthropic/OpenAI/Groq
│   ├── ethical_corpus.json        # RAG corpus v1 (21 documents)
│   ├── ethical_corpus_v2.json     # RAG corpus v2 (170 documents — recommended)
│   ├── preferences.jsonl          # Final training data (JSONL format)
│   └── synthetic_ethics.jsonl     # Generated synthetic pairs (optional)
├── training/
│   ├── train_dpo.py               # DPO with LoRA bf16 (TRL DPOTrainer)
│   ├── train_reward_model.py      # Reward model — RLHF step 1 (TRL RewardTrainer)
│   └── train_ppo.py               # PPO with bnb 4-bit — RLHF step 2 (TRL PPOTrainer)
├── eval/
│   ├── evaluate_ethics.py         # Logit-based evaluation + integrated RAG (canonical script)
│   └── plot_ppo_kl.py             # Plots the PPO KL curve (report figure)
├── notebooks/                     # Ready-to-run Kaggle notebooks (one per phase)
│   ├── 01_dpo.ipynb               # Phase 1: train DPO adapter
│   ├── 02_reward_model.ipynb      # Phase 2: train Reward Model
│   ├── 03_ppo.ipynb               # Phase 3: train PPO (attaches DPO + RM)
│   └── 04_eval.ipynb              # Phase 4: final eval on ETHICS
├── results/                       # Trained LoRA adapters, JSON scores, plots
├── report/
│   ├── main.tex                   # Report (ACL format)
│   └── references.bib
└── setup.sh                       # Linux/Kaggle setup script
```

The ethical principles corpus (`data/ethical_corpus_v2.json`, 170 documents) is loaded directly by `eval/evaluate_ethics.py`: the FAISS index is built in memory (embedder `sentence-transformers/all-MiniLM-L6-v2`) at each evaluation. No separate index-build step.

---

## Environment

| Platform | Status | Notes |
|---|---|---|
| Kaggle T4 (16 GB) | ✅ Recommended | Set `CUDA_VISIBLE_DEVICES=0` before any import |
| Linux + GPU ≥ 8 GB | ✅ OK | bnb 4-bit + bf16, Python 3.11 |
| Native Windows | ⚠️ Partial | DPO + RM OK, PPO crashes (segfault in bnb optimizers) |

**Recommended Python: 3.11** (3.12 OK — 3.13 incompatible with torch CUDA wheels and bnb).

### Installation

```bash
pip install -q -U bitsandbytes transformers==4.46.3 trl==0.12.0 peft==0.14.0 \
    accelerate==1.2.0 datasets==3.2.0 sentence-transformers faiss-cpu \
    anthropic openai pyarrow==17.0.0 tqdm
```

On Kaggle: **Restart Session** after installation (required to reload CUDA libs).

---

## Training data

| Source | Size | Role |
|---|---|---|
| PKU-SafeRLHF | ~15,000 pairs | Human safety/harmlessness preferences |
| UltraFeedback | ~5,000 pairs | High-quality general preferences |
| Synthetic (optional) | ~3,000 pairs | Moral scenarios generated via Anthropic Claude Haiku |

Final format: JSONL with `prompt`, `chosen`, `rejected`, `source` fields.

Synthetic data was integrated in a training variant but produced no measurable improvement on ETHICS — the base model's ceiling effect dominates.

**Strict constraint**: no ETHICS benchmark example is used during training.

---

## Kaggle multi-dataset workflow (recommended)

Because of Kaggle's **12-hour session limit**, the pipeline is split into **4 independent notebooks** (one per phase), with **a Kaggle Dataset exported between each step** to carry adapters from one run to the next.

The 4 ready-to-run notebooks live in `notebooks/`:

| Notebook | Datasets to attach (Add Data) | Produces | Approx. duration |
|---|---|---|---|
| `01_dpo.ipynb` | none | DPO adapter → **`adl-dpo-adapter`** | ~10 h |
| `02_reward_model.ipynb` | none | RM adapter → **`adl-reward-model`** | ~8 h |
| `03_ppo.ipynb` | **`adl-dpo-adapter` AND `adl-reward-model`** | RLHF adapter → **`adl-rlhf-model`** | ~5 h |
| `04_eval.ipynb` | `adl-dpo-adapter` + `adl-rlhf-model` | `eval_results.json` | ~30 min |

**Pattern between two notebooks**:
1. At the end of notebook N, `shutil.make_archive` zips `results/<phase>_model/` → `/kaggle/working/<phase>_model.zip`
2. Save & Run All → Output → **New Dataset** from that zip
3. At the start of notebook N+1, **Add Data** → attach the required dataset(s)
4. The first cell unzips `/kaggle/input/<dataset-name>/*.zip` into `results/<phase>_model/`

**Important for PPO**: notebook 03 needs **both** datasets (DPO + RM) attached at the same time, since PPO warm-starts its policy from the DPO adapter and uses the RM as the reward signal.

---

## Equivalent local pipeline (Linux/WSL)

If you have a Linux machine with a ≥ 16 GB GPU and want to run everything in a single session:

```bash
# 1. Data
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

# 5. Final eval
python eval/evaluate_ethics.py --dpo_adapter results/dpo_model \
    --rlhf_adapter results/rlhf_model --ethical_corpus data/ethical_corpus_v2.json \
    --output_path results/eval_results.json --n_per_cat 100

# 6. (Optional) PPO KL figure
python eval/plot_ppo_kl.py \
    --trainer_state results/rlhf_model/checkpoint-125/trainer_state.json \
    --out results/ppo_kl_divergence.png
```

---

## Hyperparameters

| Parameter | DPO | Reward Model | PPO |
|---|---|---|---|
| LoRA rank / alpha | 16 / 32 | 16 / 32 | 16 / 32 |
| Target modules | q,k,v,o | q,k,v,o | q,k,v,o |
| Dropout | 0.05 | 0.05 | 0.05 |
| Effective batch | 16 (1×16) | 16 (1×16) | 8 (2×4) |
| Learning rate | 5e-5 | 2e-5 | 1e-5 |
| β (DPO) / KL coef (PPO) | 0.1 | — | 0.05 |
| Max length (prompt + response) | 384 / 256 | 384 | 128 / 64 |
| Precision | bf16 LoRA | bf16 LoRA | bnb 4-bit base + bf16 LoRA |
| Duration | 1 epoch (1,029 steps) | 1 epoch | 1,000 episodes (125 gradient steps) |
| Warmup | 10% | — | — |
| Eval / Save | steps 200 / 100 | steps 200 / 100 | steps — / 200 |

### PPO: 4 models in memory simultaneously

PPO loads **4 instances** of the Qwen2.5-1.5B backbone on the same GPU:
1. **Policy** (warm-started from the DPO adapter, trainable)
2. **Reference** (frozen DPO adapter — KL constraint)
3. **Reward model** (frozen)
4. **Value model** (warm-started from the RM, trainable)

Under bnb 4-bit, the accumulated numerical noise across these 4 models causes explosive KL divergence (~110 vs. target < 0.5), making optimization ineffective.

---

## Evaluation

The `eval/evaluate_ethics.py` script uses a **logit-based** method without generation:

- For each example, compare `P(token "0")` vs `P(token "1")` on the prompt's last token (or `P("A")` vs `P("B")` for utilitarianism).
- The prompt goes through the Qwen chat template (`apply_chat_template`) before tokenization.
- Fast and deterministic: no sampling, no variability.
- 5 subsets × 100 examples = **500 examples total**.
- For the RAG condition: retrieve `k=3` ethical principles (cosine similarity with MiniLM, FAISS index built in RAM) and prepend them to the prompt.

Evaluated conditions: `baseline`, `dpo`, `rlhf`, `rag`.

---

## Known pitfalls

| Problem | Cause | Fix |
|---|---|---|
| PPO crashes / KL explodes | 4 models in 4-bit on 16 GB | Kaggle T4, `CUDA_VISIBLE_DEVICES=0` before any import |
| PPO segfault on Windows | bnb optimizers incompatible with Windows | Use Kaggle or Linux |
| `datasets` segfault on load | PyArrow 19+ | `pip install pyarrow==17.0.0` |
| No torch CUDA wheel | Python 3.13 | Use Python 3.11 |
| PPO sharding on Kaggle T4×2 | `accelerate` detects 2 GPUs | `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` |
| `UnicodeEncodeError` JSON | Windows cp1252 | All `open()` calls use `encoding="utf-8"` |
| OOM DPO/RM batch=4 on T4 | bf16 without bnb on Qwen 1.5B | `--batch_size 1 --grad_accum 16 --max_length 384` |
