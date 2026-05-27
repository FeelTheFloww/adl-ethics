#!/bin/bash
# ============================================================
# ADL Project — Ethical Alignment of Small Language Models
# Setup Script (Linux / Kaggle / WSL)
# ============================================================
# For local Windows: use setup.ps1 or run pip commands manually in PowerShell.
# Recommended Python: 3.11 (3.12 also works; 3.13 has wheel issues with bnb/torch CUDA).

set -e

echo "Setting up ADL project environment (Linux/Kaggle)..."

# 1. Optional: venv (skip on Kaggle, kernel already provides Python)
if [ -z "$KAGGLE_KERNEL_RUN_TYPE" ]; then
    python -m venv venv
    source venv/bin/activate
fi

pip install --upgrade pip

# 2. PyTorch (CUDA 12.1 — works on T4 and consumer GPUs; CUDA driver only needs to be >= 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Core training stack — versions pinned for reproducibility
pip install \
    transformers==4.46.3 \
    trl==0.12.0 \
    peft==0.14.0 \
    bitsandbytes==0.45.0 \
    accelerate==1.2.0 \
    datasets==3.2.0 \
    evaluate==0.4.3

# 4. RAG stack
pip install \
    sentence-transformers \
    faiss-cpu

# 5. PyArrow pinned to 17 — newer versions segfault on Kaggle/Linux when imported via datasets
pip install pyarrow==17.0.0

# 6. API clients for synthetic data generation (free tier: Groq; paid: Anthropic, OpenAI)
pip install \
    anthropic \
    openai

# 7. Utilities
pip install \
    wandb \
    scipy \
    scikit-learn \
    numpy \
    pandas \
    matplotlib \
    tqdm \
    pyyaml

echo ""
echo "Done. Verify the install:"
echo "  python -c 'import torch, bitsandbytes, transformers, trl, peft, sentence_transformers, faiss; print(\"CUDA:\", torch.cuda.is_available())'"
echo ""
echo "Next steps:"
echo "  bash run_pipeline.sh                                       # pipeline complète"
echo "  python eval/evaluate_ethics.py --skip_baseline=false       # ou uniquement le baseline"
