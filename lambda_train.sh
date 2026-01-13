#!/bin/bash
# ==============================================================================
# Epistemic Stance Classifier - Lambda Cloud Training Script
# ==============================================================================
#
# This script automates the full training pipeline on Lambda Cloud.
# 
# Usage:
#   1. Launch a Lambda Cloud instance (A10 or A100 recommended)
#   2. Upload this script and your data
#   3. Run: bash lambda_train.sh
#
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Epistemic Stance Classifier Training Pipeline"
echo "=============================================="

# Configuration
DATA_FILE="${DATA_FILE:-socialskills_multiplist_labeled.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

# Check for GPU
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. Training will be slow on CPU."
fi

# Check for data file
echo ""
echo "Checking for data file..."
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Please upload your labeled CSV file."
    exit 1
fi
echo "Found: $DATA_FILE"

# Setup virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Start training
echo ""
echo "=============================================="
echo "Starting classifier training..."
echo "=============================================="
echo "Config:"
echo "  Data: $DATA_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Epochs: $EPOCHS"
echo "  Max length: $MAX_LENGTH"
echo ""

python3 train_longformer_classifier.py \
    --data "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --epochs "$EPOCHS" \
    --max-length "$MAX_LENGTH" \
    --lr "$LEARNING_RATE" \
    --focal-loss

# Check if training succeeded
if [ ! -d "$OUTPUT_DIR/best_model" ]; then
    echo "ERROR: Training failed - no best_model found"
    exit 1
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

# Run error analysis
echo ""
echo "Running error analysis..."
if [ -f "$OUTPUT_DIR/test_predictions.csv" ]; then
    python3 error_analysis.py \
        --predictions "$OUTPUT_DIR/test_predictions.csv" \
        --output-dir "$OUTPUT_DIR/error_analysis"
fi

# Print summary
echo ""
echo "=============================================="
echo "Results Summary"
echo "=============================================="
if [ -f "$OUTPUT_DIR/results.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/results.json') as f:
    r = json.load(f)
print(f\"Best epoch: {r['best_epoch']}\")
print(f\"Test metrics:\")
for k, v in r['test_metrics'].items():
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
"
fi

echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Next steps:"
echo "=============================================="
echo "1. Review error analysis: $OUTPUT_DIR/error_analysis/"
echo "2. If satisfied, generate silver labels:"
echo "   python inference_silver_labels.py predict \\"
echo "       --model $OUTPUT_DIR/best_model \\"
echo "       --data unlabeled_data.csv \\"
echo "       --output silver_labels.csv"
echo ""
echo "3. Download results:"
echo "   scp -r ubuntu@\$(hostname):$(pwd)/$OUTPUT_DIR ./"
echo ""
