#!/bin/bash
# Fine-tune Magistral-Small-2509 using Transformers (with correct model class)
# Hardware: 2x B200 (360 GB VRAM)
# Estimated time: ~2-3 hours

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

HUB_MODEL_ID="${HUB_MODEL_ID:-}"
HUB_PRIVATE="${HUB_PRIVATE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-epistemic-stance}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

echo "=============================================="
echo "Epistemic Stance Classifier - Full Fine-tune"
echo "Model: Magistral-Small-2509 (24B)"
echo "Hardware: 2x B200 (360 GB VRAM)"
echo "=============================================="

# =============================================================================
# 1. CREATE ISOLATED ENVIRONMENT
# =============================================================================
echo "[1/6] Creating isolated virtual environment..."

rm -rf ./training_venv
python3 -m venv ./training_venv --clear
source ./training_venv/bin/activate

echo "Python: $(which python)"
pip install --upgrade pip wheel setuptools

# =============================================================================
# 2. INSTALL DEPENDENCIES (CORRECT VERSIONS FOR MAGISTRAL)
# =============================================================================
echo "[2/6] Installing dependencies..."

# Install PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install transformers with mistral-common support (as per HuggingFace docs)
# This is the key fix - we need mistral-common for proper tokenization
pip install "transformers[mistral-common]>=4.45.0"

# Verify mistral-common is installed
python -c "import mistral_common; print(f'mistral-common version: {mistral_common.__version__}')"

# Install other dependencies
pip install \
    "datasets>=2.14.0" \
    "accelerate>=0.25.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "wandb" \
    "deepspeed"

# Verify transformers can load the model class
echo "Verifying Transformers installation..."
python -c "from transformers import Mistral3ForConditionalGeneration, AutoTokenizer; print('✓ Mistral3ForConditionalGeneration available')"

# =============================================================================
# 3. HUGGINGFACE LOGIN
# =============================================================================
echo "[3/6] HuggingFace authentication..."

if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "✓ HuggingFace authenticated"
else
    echo "⚠ No HF_TOKEN found. Set it to access gated models."
fi

echo ""
echo "Make sure you've accepted the model license at:"
echo "https://huggingface.co/mistralai/Magistral-Small-2509"
echo ""

# =============================================================================
# 4. SETUP ACCELERATE
# =============================================================================
echo "[4/6] Configuring accelerate for multi-GPU..."

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

# =============================================================================
# 5. SETUP W&B (OPTIONAL)
# =============================================================================
echo "[5/6] Setting up Weights & Biases..."

if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY 2>/dev/null || true
    echo "✓ W&B configured. Project: $WANDB_PROJECT"
    WANDB_ARGS="--wandb-project $WANDB_PROJECT"
else
    echo "⚠ No WANDB_API_KEY found. Proceeding without W&B."
    WANDB_ARGS="--no-wandb"
fi

# Hub arguments
HUB_ARGS=""
if [ -n "$HUB_MODEL_ID" ]; then
    HUB_ARGS="--hub-model-id $HUB_MODEL_ID"
    [ "$HUB_PRIVATE" = "true" ] && HUB_ARGS="$HUB_ARGS --hub-private"
    echo "Will push to HuggingFace Hub: $HUB_MODEL_ID"
else
    HUB_ARGS="--no-push"
fi

# =============================================================================
# 6. RUN TRAINING
# =============================================================================
echo ""
echo "[6/6] Starting training..."
echo "=============================================="
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning rate: $LEARNING_RATE"
echo "=============================================="
echo ""
echo "This will take approximately 2-3 hours."
echo ""

accelerate launch --config_file accelerate_config.yaml finetune_epistemic_stance.py \
    --data final_training_data_balanced.csv \
    --output ./epistemic_stance_model \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRADIENT_ACCUMULATION \
    --learning-rate $LEARNING_RATE \
    $WANDB_ARGS \
    $HUB_ARGS

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved to: ./epistemic_stance_model/final"
[ -n "$HUB_MODEL_ID" ] && echo "Model pushed to: https://huggingface.co/$HUB_MODEL_ID"
echo ""
