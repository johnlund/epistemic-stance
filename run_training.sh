#!/bin/bash
# Setup and run fine-tuning on 2x B200 instance
# Estimated time: ~2-3 hours
# Estimated cost: ~$25-30

set -e

echo "=============================================="
echo "Epistemic Stance Classifier - Full Fine-tune"
echo "Model: Magistral-Small-2509 (24B)"
echo "Hardware: 2x B200 (360 GB VRAM)"
echo "=============================================="

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install --upgrade pip
pip install torch transformers datasets accelerate pandas scikit-learn wandb flash-attn --upgrade

# 2. Login to Hugging Face (for gated model access)
echo "[2/5] Hugging Face login..."
echo "Make sure you've accepted the model license at:"
echo "https://huggingface.co/mistralai/Magistral-Small-2509"
huggingface-cli login --token $HF_TOKEN

# 3. Setup accelerate config for 2 GPUs
echo "[3/5] Configuring accelerate for 2x GPU..."
cat > accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# 4. Optional: Setup W&B
echo "[4/5] Setting up Weights & Biases (optional)..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
    echo "W&B configured."
else
    echo "No WANDB_API_KEY found. Training will proceed without W&B logging."
    echo "Set WANDB_API_KEY environment variable to enable logging."
fi

# 5. Run training
echo "[5/5] Starting training..."
echo "This will take approximately 2-3 hours."
echo ""

accelerate launch --config_file accelerate_config.yaml finetune_epistemic_stance.py \
    --data final_training_data_balanced.csv \
    --output ./epistemic_stance_model \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --learning-rate 2e-5 \
    ${WANDB_API_KEY:+--wandb-project epistemic-stance} \
    ${WANDB_API_KEY:-"--no-wandb"}

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: ./epistemic_stance_model/final"
echo "=============================================="
