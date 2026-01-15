#!/bin/bash
# Setup and run fine-tuning on 2x B200 instance
# Estimated time: ~2-3 hours
# Estimated cost: ~$25-30

set -e

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

# HuggingFace Hub model ID (e.g., "username/epistemic-stance-classifier")
# Leave empty to skip pushing to Hub
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

# Make the Hub repo private? (set to "true" or "false")
HUB_PRIVATE="${HUB_PRIVATE:-false}"

# W&B project name
WANDB_PROJECT="${WANDB_PROJECT:-epistemic-stance}"

# Training parameters
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

# =============================================================================
# SETUP
# =============================================================================

echo "=============================================="
echo "Epistemic Stance Classifier - Full Fine-tune"
echo "Model: Magistral-Small-2509 (24B)"
echo "Hardware: 2x B200 (360 GB VRAM)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning rate: $LEARNING_RATE"
echo "  Hub model ID: ${HUB_MODEL_ID:-'(not pushing to Hub)'}"
echo "  W&B project: ${WANDB_API_KEY:+$WANDB_PROJECT}${WANDB_API_KEY:-'(disabled)'}"
echo ""

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install --upgrade pip
# Install all packages together with constraints so pip can resolve dependencies properly
# Pin numpy to <1.25.0 for compatibility with system scipy
# Pin huggingface_hub to <1.0 for compatibility with transformers 4.44.2
pip install \
    "numpy>=1.17.3,<1.25.0" \
    "huggingface_hub<1.0,>=0.34.0" \
    torch \
    "transformers==4.44.2" \
    datasets \
    accelerate \
    pandas \
    scikit-learn \
    wandb \
    tf-keras \
    --upgrade
# Using SDPA attention (built into PyTorch) instead of flash-attn to avoid compilation

# 2. Login to Hugging Face (for gated model access and upload)
echo "[2/5] Hugging Face login..."
if [ -n "$HF_TOKEN" ]; then
    # Try huggingface-cli first, fall back to python -m if not in PATH
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli login --token $HF_TOKEN
    else
        python3 -m huggingface_hub.cli.login --token $HF_TOKEN
    fi
    echo "HuggingFace authenticated."
else
    echo "WARNING: No HF_TOKEN found. You may not be able to access gated models."
    echo "Set HF_TOKEN environment variable if you encounter access issues."
fi

echo ""
echo "Make sure you've accepted the model license at:"
echo "https://huggingface.co/mistralai/Magistral-Small-2509"
echo ""

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
echo "[4/5] Setting up Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
    echo "W&B configured. Project: $WANDB_PROJECT"
    WANDB_ARGS="--wandb-project $WANDB_PROJECT"
else
    echo "No WANDB_API_KEY found. Training will proceed without W&B logging."
    echo "Set WANDB_API_KEY environment variable to enable logging."
    WANDB_ARGS="--no-wandb"
fi

# 5. Build Hub arguments
HUB_ARGS=""
if [ -n "$HUB_MODEL_ID" ]; then
    HUB_ARGS="--hub-model-id $HUB_MODEL_ID"
    if [ "$HUB_PRIVATE" = "true" ]; then
        HUB_ARGS="$HUB_ARGS --hub-private"
    fi
    echo "Will push to HuggingFace Hub: $HUB_MODEL_ID"
else
    HUB_ARGS="--no-push"
    echo "Not pushing to HuggingFace Hub (set HUB_MODEL_ID to enable)"
fi

# 6. Run training
echo ""
echo "[5/5] Starting training..."
echo "This will take approximately 2-3 hours."
echo ""

# Disable TensorFlow imports since we're only using PyTorch
export TRANSFORMERS_NO_TF=1

accelerate launch --config_file accelerate_config.yaml finetune_epistemic_stance.py \
    --data final_training_data_balanced.csv \
    --output ./epistemic_stance_model \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRADIENT_ACCUMULATION \
    --learning-rate $LEARNING_RATE \
    $WANDB_ARGS \
    $HUB_ARGS

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved to: ./epistemic_stance_model/final"
if [ -n "$HUB_MODEL_ID" ]; then
    echo "Model pushed to: https://huggingface.co/$HUB_MODEL_ID"
fi
echo ""
echo "To run inference:"
echo "  python inference_epistemic_stance.py --model ./epistemic_stance_model/final --interactive"
echo ""
