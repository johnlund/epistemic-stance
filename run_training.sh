#!/bin/bash
# Setup and run fine-tuning on 2x B200 instance
# Estimated time: ~2-3 hours
# Estimated cost: ~$25-30

set -e

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

HUB_MODEL_ID="${HUB_MODEL_ID:-}"
HUB_PRIVATE="${HUB_PRIVATE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-epistemic-stance}"

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

# =============================================================================
# CRITICAL: Disable TensorFlow/Keras imports BEFORE any Python runs
# =============================================================================
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export USE_TF=0
export USE_TORCH=1

# =============================================================================
# 1. Create isolated virtual environment
# =============================================================================
echo "[1/6] Creating isolated virtual environment..."

# Remove any existing venv
rm -rf ./training_venv

# Create fresh venv WITHOUT system site-packages to avoid conflicts
python3 -m venv ./training_venv --clear

# Activate it
source ./training_venv/bin/activate

# Verify we're in the venv
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# =============================================================================
# 2. Install dependencies in correct order
# =============================================================================
echo "[2/6] Installing dependencies..."

# Upgrade pip first
pip install --upgrade pip wheel setuptools

# Install PyTorch first (foundation for everything else)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install core ML packages with compatible versions
# Key: transformers 4.44.2 is stable and doesn't have the PreTrainedModel import bug
pip install \
    "transformers==4.44.2" \
    "huggingface_hub>=0.23.0,<0.25.0" \
    "datasets>=2.14.0" \
    "accelerate>=0.25.0" \
    "safetensors>=0.4.0" \
    "tokenizers>=0.15.0"

# Install data science packages
pip install \
    "numpy>=1.24.0,<2.0.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0"

# Install optional packages
pip install wandb

# Verify transformers works
echo "Verifying transformers installation..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments; print('✓ Transformers OK')"

# =============================================================================
# 3. HuggingFace login
# =============================================================================
echo "[3/6] HuggingFace login..."
if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "✓ HuggingFace authenticated"
else
    echo "⚠ No HF_TOKEN found. Set it if you need to access gated models."
fi

echo ""
echo "Make sure you've accepted the model license at:"
echo "https://huggingface.co/mistralai/Magistral-Small-2509"
echo ""

# =============================================================================
# 4. Setup accelerate for multi-GPU
# =============================================================================
echo "[4/6] Configuring accelerate for 2x GPU..."

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
# 5. Setup W&B (optional)
# =============================================================================
echo "[5/6] Setting up Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY 2>/dev/null || true
    echo "✓ W&B configured. Project: $WANDB_PROJECT"
    WANDB_ARGS="--wandb-project $WANDB_PROJECT"
else
    echo "⚠ No WANDB_API_KEY found. Training will proceed without W&B logging."
    WANDB_ARGS="--no-wandb"
fi

# Build Hub arguments
HUB_ARGS=""
if [ -n "$HUB_MODEL_ID" ]; then
    HUB_ARGS="--hub-model-id $HUB_MODEL_ID"
    [ "$HUB_PRIVATE" = "true" ] && HUB_ARGS="$HUB_ARGS --hub-private"
    echo "Will push to HuggingFace Hub: $HUB_MODEL_ID"
else
    HUB_ARGS="--no-push"
fi

# =============================================================================
# 6. Run training
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
echo "To run inference:"
echo "  source ./training_venv/bin/activate"
echo "  python inference_epistemic_stance.py --model ./epistemic_stance_model/final --interactive"
echo ""
