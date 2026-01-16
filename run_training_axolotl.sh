#!/bin/bash
# Fine-tune Magistral-Small-2509 using Axolotl (officially recommended by Mistral)
# Hardware: 4x H100 80GB SXM5 (320 GB VRAM) - Full fine-tune
# Estimated time: ~2 hours
# Estimated cost: ~$25 ($12.36/hr)

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

HUB_MODEL_ID="${HUB_MODEL_ID:-}"
WANDB_PROJECT="${WANDB_PROJECT:-epistemic-stance}"
EPOCHS="${EPOCHS:-3}"

echo "=============================================="
echo "Epistemic Stance Classifier - Axolotl Training"
echo "Model: Magistral-Small-2509 (24B)"
echo "Hardware: 4x H100 80GB SXM5 (Full fine-tune)"
echo "=============================================="

# =============================================================================
# 1. SETUP ENVIRONMENT
# =============================================================================
echo "[1/5] Setting up environment..."

# Create and activate virtual environment
rm -rf ./axolotl_venv
# python3 -m venv ./axolotl_venv --clear
source ./axolotl_venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# =============================================================================
# 2. INSTALL AXOLOTL
# =============================================================================
echo "[2/5] Installing Axolotl and dependencies..."

# Install PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install Axolotl with all dependencies
pip install axolotl[deepspeed]

# Install additional requirements
pip install pandas wandb

# Verify installation
python -c "import axolotl; print(f'Axolotl version: {axolotl.__version__}')"

# =============================================================================
# 3. HUGGINGFACE LOGIN
# =============================================================================
echo "[3/5] HuggingFace authentication..."

if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "✓ HuggingFace authenticated"
else
    echo "⚠ No HF_TOKEN found. Set it to access gated models."
    exit 1
fi

echo ""
echo "Make sure you've accepted the model license at:"
echo "https://huggingface.co/mistralai/Magistral-Small-2509"
echo ""

# =============================================================================
# 4. PREPARE DATA
# =============================================================================
echo "[4/5] Preparing training data..."

python prepare_data_for_axolotl.py \
    --input final_training_data_balanced.csv \
    --output final_training_data_formatted.jsonl

# Create deepspeed config directory if needed
mkdir -p deepspeed_configs

# Copy deepspeed config if not present
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

# Update config with optional settings
if [ -n "$HUB_MODEL_ID" ]; then
    echo "Will push to HuggingFace Hub: $HUB_MODEL_ID"
    # Append hub settings to config
    cat >> axolotl_config.yaml << EOF

# Hub settings (dynamically added)
hub_model_id: $HUB_MODEL_ID
push_to_hub: true
EOF
fi

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
echo "  Model: Magistral-Small-2509 (24B)"
echo "  Method: Full fine-tune with DeepSpeed ZeRO-2"
echo "  Hardware: 4x H100 80GB (320GB total)"
echo "=============================================="
echo ""
echo "This will take approximately 2 hours."
echo ""

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
