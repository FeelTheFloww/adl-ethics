#!/bin/bash
# ============================================================
# ADL Project - Ethical Alignment of Small Language Models
# Setup Script
# ============================================================

set -e

echo "Setting up ADL project environment..."

# 1. Virtual env
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. PyTorch (CUDA 12.1 for RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Core training / inference stack
pip install \
    transformers==4.47.0 \
    trl==0.12.0 \
    peft==0.14.0 \
    bitsandbytes==0.45.0 \
    accelerate==1.2.0 \
    datasets==3.2.0 \
    evaluate==0.4.3

# 4. RAG stack
pip install \
    sentence-transformers==3.3.1 \
    faiss-cpu==1.9.0

# 5. Optional: API clients for synthetic data generation
pip install \
    anthropic==0.39.0 \
    openai==1.54.0

# 6. Utility
pip install \
    wandb \
    scipy \
    scikit-learn \
    numpy \
    pandas \
    matplotlib \
    tqdm \
    pyyaml

echo "Done."
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  wandb login"
echo "  python eval/evaluate_baseline.py --model_name Qwen/Qwen2.5-1.5B-Instruct"
