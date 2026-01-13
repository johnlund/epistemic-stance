# Epistemic Stance Classifier

A Longformer-based classifier for detecting epistemic stance (absolutist, multiplist, evaluativist) in text, based on Kuhn et al.'s developmental epistemology framework.

## Overview

This pipeline trains a classifier to identify three epistemic stances:

- **Absolutist**: Knowledge as certain facts; claims presented without qualification; counterarguments dismissed
- **Multiplist**: Knowledge as subjective opinion; multiple perspectives acknowledged but not evaluated; "everyone's entitled to their view"
- **Evaluativist**: Knowledge as reasoned judgment; evidence weighed; counterarguments engaged substantively

## Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Gold Labels       │     │   Classifier        │     │   Silver Labels     │
│   (2,806 samples)   │ ──▶ │   (Longformer)      │ ──▶ │   (~28,000 samples) │
│   LLM + Review      │     │   Train & Eval      │     │   High-confidence   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                  │
                                                                  ▼
                                                        ┌─────────────────────┐
                                                        │   Combined Data     │
                                                        │   Gold + Silver     │
                                                        │   For Fine-tuning   │
                                                        └─────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Classifier

```bash
python train_longformer_classifier.py \
    --data socialskills_multiplist_labeled.csv \
    --output-dir ./classifier_output \
    --focal-loss \
    --epochs 5
```

### 3. Generate Silver Labels

```bash
python inference_silver_labels.py predict \
    --model ./classifier_output/best_model \
    --data unlabeled_socialskills.csv \
    --output silver_labels.csv \
    --threshold 0.8
```

### 4. Combine for Fine-tuning

```bash
python inference_silver_labels.py combine \
    --gold socialskills_multiplist_labeled.csv \
    --silver silver_labels.csv \
    --output combined_training_data.csv \
    --gold-weight 2.0
```

## Lambda Cloud Setup

### Recommended Instance

For training the Longformer classifier:
- **GPU**: 1x A10 (24GB) or 1x A100 (40GB)
- **Estimated cost**: $0.60-1.10/hr
- **Estimated training time**: 2-4 hours

### Setup Commands

```bash
# SSH into Lambda instance
ssh ubuntu@<your-instance-ip>

# Clone/upload your code
git clone <your-repo> epistemic_classifier
cd epistemic_classifier

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Upload your data (from local machine)
scp socialskills_multiplist_labeled.csv ubuntu@<ip>:~/epistemic_classifier/

# Run training
python train_longformer_classifier.py \
    --data socialskills_multiplist_labeled.csv \
    --output-dir ./output \
    --focal-loss \
    --epochs 5 \
    --batch-size 4 \
    --grad-accum 4

# Download results (from local machine)
scp -r ubuntu@<ip>:~/epistemic_classifier/output ./
```

### Memory Optimization

If you encounter OOM errors:

```bash
# Reduce batch size and increase gradient accumulation
python train_longformer_classifier.py \
    --data data.csv \
    --batch-size 2 \
    --grad-accum 8 \
    --max-length 1536  # Reduce if needed
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to labeled CSV |
| `--output-dir` | ./output | Where to save model |
| `--model` | longformer-base-4096 | Base model |
| `--max-length` | 2048 | Max sequence length |
| `--batch-size` | 4 | Training batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 2e-5 | Learning rate |
| `--epochs` | 5 | Number of epochs |
| `--focal-loss` | False | Use focal loss |
| `--focal-gamma` | 2.0 | Focal loss gamma |
| `--no-class-weights` | False | Disable class weights |
| `--no-calibration` | False | Disable temperature scaling |

## Expected Results

Based on the data distribution:
- **Evaluativist**: 67.6% (1,896 samples)
- **Absolutist**: 27.1% (761 samples)
- **Multiplist**: 5.3% (149 samples)

Target metrics:
- Overall accuracy: >85%
- F1-macro: >0.80
- F1-multiplist: >0.60 (challenging due to small sample size)
- Calibration ECE: <0.10

## Output Files

After training, the output directory contains:

```
output/
├── best_model/              # Best checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── temperature_scaler.pt
├── checkpoint-epoch-*/      # Epoch checkpoints
├── train_split.csv          # Training data split
├── val_split.csv            # Validation split
├── test_split.csv           # Test split
├── test_predictions.csv     # Predictions with probabilities
├── results.json             # All metrics and config
└── training.log             # Training logs
```

## Confidence Filtering for Silver Labels

The inference script filters predictions by confidence:

- **Very High (>0.9)**: High-quality silver labels
- **High (0.8-0.9)**: Good quality, recommended threshold
- **Medium (0.6-0.8)**: Use with caution
- **Low (<0.6)**: Exclude from training

Recommended: Use threshold of 0.8 for silver labels.

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce memory usage
--batch-size 2 --grad-accum 8 --max-length 1536
```

### Low Multiplist F1

The multiplist class has only 149 samples. If F1 is very low:
1. Increase focal loss gamma: `--focal-gamma 3.0`
2. Manually review multiplist samples for labeling consistency
3. Consider data augmentation (paraphrasing)

### Poor Calibration

If ECE > 0.15 after temperature scaling:
1. Check if validation set is representative
2. Try training longer
3. Consider Platt scaling instead

## Citation

Based on:
- Kuhn, D., Cheney, R., & Weinstock, M. (2000). The development of epistemological understanding.
- Nussbaum, E. M., Sinatra, G. M., & Poliquin, A. (2008). Role of epistemic beliefs and scientific argumentation in science learning.

## License

MIT License - See LICENSE file for details.
