#!/bin/bash
# Fine-tune Mistral-Small-24B-Instruct-2501 using Axolotl
# Hardware: 4x H100 80GB SXM5 (320 GB VRAM) - Full fine-tune
# Estimated time: ~2 hours
# Estimated cost: ~$25 ($12.36/hr)
#
# Mistral-Small-24B-Instruct-2501 (Mistral Small 3):
# - 24B parameters, text-only
# - Tekken V7 tokenizer (131k vocab)
# - 32k context window
# - Apache 2.0 license (no gated access required)

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

HUB_MODEL_ID="${HUB_MODEL_ID:-}"
WANDB_PROJECT="${WANDB_PROJECT:-epistemic-stance}"
EPOCHS="${EPOCHS:-3}"

echo "=============================================="
echo "Epistemic Stance Classifier - Axolotl Training"
echo "Model: Mistral-Small-24B-Instruct-2501 (24B)"
echo "Hardware: 4x H100 80GB SXM5 (Full fine-tune)"
echo "=============================================="

# =============================================================================
# 1. SETUP ENVIRONMENT
# =============================================================================
echo "[1/5] Setting up environment..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d "./axolotl_venv" ]; then
    python3 -m venv ./axolotl_venv
fi
source ./axolotl_venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# =============================================================================
# 2. INSTALL AXOLOTL AND DEPENDENCIES
# =============================================================================
echo "[2/5] Installing Axolotl and dependencies..."

# Install PyTorch first (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install Axolotl with all dependencies
pip install axolotl[deepspeed]

# Install Liger kernel for memory optimization (recommended for Mistral)
pip install liger-kernel

# Install additional requirements
pip install pandas wandb mistral_common

# Verify installation
python -c "import axolotl; print(f'Axolotl version: {axolotl.__version__}')"
python -c "import liger_kernel; print('Liger kernel installed successfully')"

# =============================================================================
# 3. HUGGINGFACE LOGIN (Optional - Apache 2.0 license, no gating)
# =============================================================================
echo "[3/5] HuggingFace authentication..."

if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "✓ HuggingFace authenticated"
else
    echo "⚠ No HF_TOKEN found. This is optional for Mistral-Small-24B-Instruct-2501"
    echo "  (Apache 2.0 license - no gated access required)"
fi

echo ""
echo "Model: mistralai/Mistral-Small-24B-Instruct-2501"
echo "License: Apache 2.0 (open access)"
echo ""

# =============================================================================
# 4. PREPARE DATA
# =============================================================================
echo "[4/5] Preparing training data..."

# Only run data prep if the output doesn't exist
if [ ! -f "final_training_data_formatted.jsonl" ]; then
    python prepare_data_for_axolotl.py \
        --input final_training_data_balanced.csv \
        --output final_training_data_formatted.jsonl
else
    echo "Using existing final_training_data_formatted.jsonl"
fi

# Create deepspeed config directory if needed
mkdir -p deepspeed_configs

# Create DeepSpeed ZeRO-2 config (optimal for 4x H100 full fine-tune)
if [ ! -f deepspeed_configs/zero2.json ]; then
    cat > deepspeed_configs/zero2.json << 'DSEOF'
{
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
DSEOF
fi

# =============================================================================
# 5. UPDATE CONFIG AND RUN TRAINING
# =============================================================================
echo "[5/5] Starting training..."

if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY 2>/dev/null || true
    echo "✓ W&B configured"
fi

# Update epochs in config
sed -i "s/num_epochs: .*/num_epochs: $EPOCHS/" axolotl_config.yaml

echo ""
echo "=============================================="
echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Model: Mistral-Small-24B-Instruct-2501 (24B)"
echo "  Method: Full fine-tune with DeepSpeed ZeRO-2"
echo "  Hardware: 4x H100 80GB (320GB total)"
echo "  Optimizations: Liger kernel, Flash Attention"
echo "=============================================="
echo ""
echo "This will take approximately 2 hours."
echo ""

# Clear any cached prepared data that might have stale settings
rm -rf ./prepared_data

# Run Axolotl training with 4 GPUs
accelerate launch --num_processes 4 --mixed_precision bf16 -m axolotl.cli.train axolotl_config.yaml

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved to: ./epistemic_stance_model"
[ -n "$HUB_MODEL_ID" ] && echo "Model pushed to: https://huggingface.co/$HUB_MODEL_ID"
echo ""
echo "To run inference:"
echo "  source ./axolotl_venv/bin/activate"
echo "  python inference_epistemic_stance.py --model ./epistemic_stance_model --interactive"
echo ""
