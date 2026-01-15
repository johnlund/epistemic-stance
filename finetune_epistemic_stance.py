#!/usr/bin/env python3
"""
Full fine-tune Magistral-Small-2509 (24B) for epistemic stance classification.

Optimized for 2x B200 (360 GB VRAM total).

Usage:
    # Single command to run training
    accelerate launch --num_processes 2 --mixed_precision bf16 finetune_epistemic_stance.py \
        --data final_training_data_balanced.csv \
        --output ./epistemic_stance_model \
        --epochs 3 \
        --hub-model-id your-username/epistemic-stance-analyzer

Requirements:
    pip install torch transformers datasets accelerate pandas scikit-learn wandb evaluate
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "mistralai/Magistral-Small-2509"
MAX_LENGTH = 2048  # Max sequence length for training

# Label mapping for metrics
LABEL_TO_ID = {"absolutist": 0, "multiplist": 1, "evaluativist": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# System prompt for epistemic stance classification
SYSTEM_PROMPT = """You are an expert classifier trained to identify epistemic stances in text based on Kuhn's developmental epistemology framework.

## The Three Epistemic Stances

**ABSOLUTIST**
Core belief: Knowledge is CERTAIN. There is ONE RIGHT ANSWER.

How they argue:
- Present claims as objective truth, not as a perspective
- Dismiss opposing views as wrong or misinformed
- Use evidence as PROOF, not as support for judgment
- Show little genuine engagement with counterarguments

Justification pattern: By authority or "obviousness" — no reasoning deemed necessary.

Linguistic markers: "Obviously...", "The truth is...", "You need to...", "The only option is...", "Anyone who thinks X is wrong", "This proves..."

Example: "You need to leave him. This is a huge red flag and staying would be a mistake. Don't let him manipulate you."

---

**MULTIPLIST**
Core belief: Knowledge is SUBJECTIVE. All opinions are EQUALLY VALID.

How they argue:
- Frame claims as personal opinions without justification
- Avoid evaluating which perspective has more merit
- Treat disagreement as natural and unresolvable
- Present multiple views but refuse to weigh them

Justification pattern: By personal preference — "that's just how I feel."

Linguistic markers: "That's just my opinion", "Everyone's entitled to their view", "Who's to say what's right?", "It depends on the person", "Only you can know"

Example: "Some people think you should leave, others think you should stay. Only you know what's right for you."

---

**EVALUATIVIST**
Core belief: Knowledge is UNCERTAIN but some claims have MORE MERIT based on evidence and reasoning.

How they argue:
- Acknowledge their position is a judgment, not absolute truth
- Engage substantively with counterarguments
- Weigh competing perspectives and explain why one is more compelling
- Show calibrated confidence matched to complexity

Justification pattern: Multiple justifications, cross-checked — cites reasoning AND evidence, acknowledges limitations.

Linguistic markers: "The evidence suggests...", "On balance...", "I could be wrong, but...", "While I understand X, I find Y more compelling because...", "Based on what you've described..."

Example: "Based on what you've described, the pattern suggests he's not respecting your boundaries. I'd lean toward reconsidering the relationship, though you know details I don't."

---

## Key Distinctions

**Absolutist vs. Evaluativist**: Both take positions. Ask: Do they JUSTIFY with reasoning, or assert as fact?
- Absolutist: "You must leave him" (directive, no reasoning)
- Evaluativist: "Based on the pattern you described, leaving seems stronger because..." (reasoned)

**Multiplist vs. Evaluativist**: Both acknowledge perspectives. Ask: Do they WEIGH them?
- Multiplist: "Some say leave, some say stay. Only you know." (no weighing)
- Evaluativist: "Leaving protects your health; staying might work if he changes. I'd lean toward leaving." (weighs tradeoffs)

**Deference distinction**: 
- Multiplist defers INSTEAD OF evaluation: "Only you can decide."
- Evaluativist defers AFTER evaluation: "I'd lean toward X, but you know things I don't."

---

Respond with JSON: {"stance": "absolutist|multiplist|evaluativist", "confidence": "high|medium|low"}"""

# =============================================================================
# DATA PREPARATION
# =============================================================================

def format_training_example(text: str, label: str, confidence: str = "high") -> dict:
    """Format a single training example as a conversation."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the epistemic stance in the following text:\n\n{text}"},
            {"role": "assistant", "content": json.dumps({"stance": label, "confidence": confidence})}
        ],
        "label": label,  # Keep label for stratification and metrics
    }


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.1,
    seed: int = 42
) -> tuple[Dataset, Dataset, dict]:
    """Load CSV and prepare train/eval datasets."""
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_cols = ['text', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Log distribution
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Compute dataset statistics for logging
    dataset_stats = {
        "total_samples": len(df),
        "label_distribution": df['label'].value_counts().to_dict(),
        "data_source_distribution": df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {},
        "avg_text_length": df['text'].str.len().mean(),
        "median_text_length": df['text'].str.len().median(),
    }
    
    # Map confidence based on sample weight (heuristic)
    def get_confidence(row):
        weight = row.get('sample_weight', 1.0)
        if weight >= 2.0:
            return "high"  # Gold data
        elif weight >= 1.0:
            return "medium"  # Silver data
        else:
            return "low"
    
    # Format examples
    examples = []
    for _, row in df.iterrows():
        confidence = get_confidence(row)
        example = format_training_example(
            text=row['text'],
            label=row['label'],
            confidence=confidence
        )
        example['sample_weight'] = row.get('sample_weight', 1.0)
        example['text_original'] = row['text']  # Keep for generation eval
        examples.append(example)
    
    # Split train/eval
    train_examples, eval_examples = train_test_split(
        examples,
        test_size=test_size,
        random_state=seed,
        stratify=[e['label'] for e in examples]
    )
    
    logger.info(f"Train samples: {len(train_examples)}")
    logger.info(f"Eval samples: {len(eval_examples)}")
    
    dataset_stats["train_samples"] = len(train_examples)
    dataset_stats["eval_samples"] = len(eval_examples)
    
    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples), dataset_stats


def tokenize_conversation(example: dict, tokenizer) -> dict:
    """Tokenize a conversation for causal LM training."""
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )
    
    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    # Mask system and user tokens (only train on assistant response)
    # Find where assistant response starts by looking for the assistant header in the formatted text
    # Mistral uses [INST]...[/INST] format, so we find the last [/INST] marker
    input_ids = tokenized['input_ids']
    labels = tokenized['labels']
    
    # Try multiple possible markers for assistant response start
    possible_markers = ["[/INST]", "</s>", "<|assistant|>"]
    mask_until = 0
    
    for marker in possible_markers:
        marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
        if not marker_tokens:
            continue
        for i in range(len(input_ids) - len(marker_tokens)):
            if input_ids[i:i+len(marker_tokens)] == marker_tokens:
                mask_until = i + len(marker_tokens)
        if mask_until > 0:
            break
    
    # Mask everything before assistant response with -100
    for i in range(mask_until):
        labels[i] = -100
    
    tokenized['labels'] = labels
    
    # Store label_id for metrics computation
    tokenized['label_id'] = LABEL_TO_ID.get(example['label'], -1)
    
    return tokenized


# =============================================================================
# METRICS
# =============================================================================

def compute_classification_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Compute classification metrics from string predictions.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Filter valid predictions
    valid_labels = set(LABEL_TO_ID.keys())
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p in valid_labels and r in valid_labels]
    
    if not valid_pairs:
        return {"accuracy": 0.0, "valid_predictions": 0}
    
    preds, refs = zip(*valid_pairs)
    
    metrics = {
        "accuracy": accuracy_score(refs, preds),
        "f1_macro": f1_score(refs, preds, average='macro', zero_division=0),
        "f1_weighted": f1_score(refs, preds, average='weighted', zero_division=0),
        "valid_predictions": len(valid_pairs),
        "invalid_predictions": len(predictions) - len(valid_pairs),
    }
    
    # Per-class metrics
    for label in valid_labels:
        label_preds = [1 if p == label else 0 for p in preds]
        label_refs = [1 if r == label else 0 for r in refs]
        
        metrics[f"precision_{label}"] = precision_score(label_refs, label_preds, zero_division=0)
        metrics[f"recall_{label}"] = recall_score(label_refs, label_preds, zero_division=0)
        metrics[f"f1_{label}"] = f1_score(label_refs, label_preds, zero_division=0)
    
    return metrics


# =============================================================================
# CUSTOM TRAINER WITH SAMPLE WEIGHTS
# =============================================================================

class WeightedTrainer(Trainer):
    """Trainer that supports sample weights in loss computation."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss."""
        
        # Extract sample weights if present
        sample_weights = inputs.pop('sample_weight', None)
        # Remove label_id - not needed for forward pass
        inputs.pop('label_id', None)
        
        # Forward pass
        outputs = model(**inputs)
        
        if sample_weights is not None and self.args.label_smoothing_factor == 0:
            # Get per-token losses
            logits = outputs.logits
            labels = inputs['labels']
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute per-token cross entropy
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            per_token_loss = loss_fct(flat_logits, flat_labels)
            per_token_loss = per_token_loss.view(shift_labels.size())
            
            # Mask padding and compute per-sample loss
            mask = (shift_labels != -100).float()
            per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            # Apply sample weights
            sample_weights = sample_weights.to(per_sample_loss.device)
            weighted_loss = (per_sample_loss * sample_weights).mean()
            
            loss = weighted_loss
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# EVALUATION WITH GENERATION
# =============================================================================

def evaluate_with_generation(
    model,
    tokenizer,
    eval_dataset,
    num_samples: int = 200,
    max_new_tokens: int = 50
) -> dict:
    """
    Run generation-based evaluation on a subset of eval data.
    Returns classification metrics.
    """
    import random
    
    logger.info(f"Running generation-based evaluation on {num_samples} samples...")
    
    # Sample subset
    indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            example = eval_dataset[idx]
            
            # Get reference label
            ref_label = example.get('label', 'unknown')
            references.append(ref_label)
            
            # Get original text
            text = example.get('text_original', '')
            
            # Build prompt (without assistant response)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify the epistemic stance in the following text:\n\n{text}"}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Parse prediction
            try:
                result = json.loads(response)
                pred_label = result.get('stance', 'unknown')
            except json.JSONDecodeError:
                # Try to extract from malformed response
                pred_label = 'unknown'
                for label in LABEL_TO_ID.keys():
                    if label in response.lower():
                        pred_label = label
                        break
            
            predictions.append(pred_label)
    
    # Compute metrics
    metrics = compute_classification_metrics(predictions, references)
    
    return metrics


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Magistral for epistemic stance classification')
    parser.add_argument('--data', '-d', required=True, help='Path to training CSV')
    parser.add_argument('--output', '-o', required=True, help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Per-device batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--test-size', type=float, default=0.1, help='Eval set fraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # W&B arguments
    parser.add_argument('--wandb-project', type=str, default='epistemic-stance', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    # HuggingFace Hub arguments
    parser.add_argument('--hub-model-id', type=str, default=None, 
                        help='HuggingFace Hub model ID (e.g., username/model-name)')
    parser.add_argument('--hub-private', action='store_true', help='Make the Hub repo private')
    parser.add_argument('--no-push', action='store_true', help='Disable pushing to Hub')
    
    args = parser.parse_args()
    
    # ==========================================================================
    # SETUP W&B
    # ==========================================================================
    
    if not args.no_wandb:
        try:
            import wandb
            
            wandb_run_name = args.wandb_run_name or f"epistemic-stance-magistral-{args.epochs}ep"
            
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config={
                    "model_id": MODEL_ID,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "learning_rate": args.learning_rate,
                    "warmup_ratio": args.warmup_ratio,
                    "max_length": MAX_LENGTH,
                    "test_size": args.test_size,
                    "seed": args.seed,
                }
            )
            logger.info(f"W&B initialized: {wandb.run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            args.no_wandb = True
    
    if args.no_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # ==========================================================================
    # LOAD TOKENIZER
    # ==========================================================================
    
    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ==========================================================================
    # LOAD AND PREPARE DATA
    # ==========================================================================
    
    train_dataset, eval_dataset, dataset_stats = load_and_prepare_data(
        args.data,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Log dataset stats to W&B
    if not args.no_wandb:
        try:
            import wandb
            wandb.config.update({"dataset": dataset_stats})
            
            # Log label distribution as a bar chart
            label_dist = dataset_stats.get("label_distribution", {})
            if label_dist:
                wandb.log({
                    "label_distribution": wandb.plot.bar(
                        wandb.Table(
                            data=[[k, v] for k, v in label_dist.items()],
                            columns=["label", "count"]
                        ),
                        "label", "count",
                        title="Training Label Distribution"
                    )
                })
        except Exception as e:
            logger.warning(f"Failed to log dataset stats to W&B: {e}")
    
    # ==========================================================================
    # TOKENIZE DATASETS
    # ==========================================================================
    
    logger.info("Tokenizing datasets...")
    
    # Keep a copy of eval dataset for generation-based evaluation
    eval_dataset_raw = eval_dataset
    
    def tokenize_fn(example):
        tokenized = tokenize_conversation(example, tokenizer)
        tokenized['sample_weight'] = example['sample_weight']
        return tokenized
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        remove_columns=['messages', 'label', 'text_original'],
        num_proc=8,
        desc="Tokenizing train"
    )
    
    eval_dataset_tokenized = eval_dataset.map(
        tokenize_fn,
        remove_columns=['messages'],  # Keep 'label' and 'text_original' for gen eval
        num_proc=8,
        desc="Tokenizing eval"
    )
    
    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================
    
    logger.info(f"Loading model from {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Calculate effective batch size
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = args.batch_size * args.gradient_accumulation * num_gpus
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # ==========================================================================
    # TRAINING ARGUMENTS
    # ==========================================================================
    
    # Determine if we should push to hub
    push_to_hub = args.hub_model_id is not None and not args.no_push
    
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        
        # Precision
        bf16=True,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        
        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Performance
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        
        # Distributed
        ddp_find_unused_parameters=False,
        
        # HuggingFace Hub
        push_to_hub=push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        hub_private_repo=args.hub_private,
        
        # Misc
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.wandb_run_name or f"epistemic-stance-magistral-{args.epochs}ep",
    )
    
    # ==========================================================================
    # DATA COLLATOR
    # ==========================================================================
    
    # Custom collator that handles extra fields
    class CustomDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            # Separate out fields that shouldn't go to the model
            sample_weights = [f.pop('sample_weight', 1.0) for f in features]
            label_ids = [f.pop('label_id', -1) for f in features]
            
            # Remove any leftover string fields
            for f in features:
                f.pop('label', None)
                f.pop('text_original', None)
            
            # Call parent collator
            batch = super().__call__(features)
            
            # Add back sample weights as tensor
            batch['sample_weight'] = torch.tensor(sample_weights, dtype=torch.float32)
            batch['label_id'] = torch.tensor(label_ids, dtype=torch.long)
            
            return batch
    
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # ==========================================================================
    # INITIALIZE TRAINER
    # ==========================================================================
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_tokenized,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # ==========================================================================
    # TRAIN
    # ==========================================================================
    
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Log training metrics
    logger.info(f"Training metrics: {train_result.metrics}")
    
    if not args.no_wandb:
        try:
            import wandb
            wandb.log({f"train/{k}": v for k, v in train_result.metrics.items()})
        except Exception:
            pass
    
    # ==========================================================================
    # FINAL EVALUATION WITH GENERATION
    # ==========================================================================
    
    logger.info("Running final generation-based evaluation...")
    
    gen_metrics = evaluate_with_generation(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset_raw,
        num_samples=min(200, len(eval_dataset_raw))
    )
    
    logger.info(f"Generation-based evaluation results:")
    for k, v in gen_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    
    # Log final metrics to W&B
    if not args.no_wandb:
        try:
            import wandb
            wandb.log({f"final/{k}": v for k, v in gen_metrics.items()})
            
            # Create summary table
            wandb.run.summary["final_accuracy"] = gen_metrics.get("accuracy", 0)
            wandb.run.summary["final_f1_macro"] = gen_metrics.get("f1_macro", 0)
            wandb.run.summary["final_f1_weighted"] = gen_metrics.get("f1_weighted", 0)
            
            # Per-class F1
            for label in LABEL_TO_ID.keys():
                key = f"f1_{label}"
                if key in gen_metrics:
                    wandb.run.summary[f"final_{key}"] = gen_metrics[key]
            
        except Exception as e:
            logger.warning(f"Failed to log final metrics to W&B: {e}")
    
    # ==========================================================================
    # SAVE FINAL MODEL
    # ==========================================================================
    
    logger.info(f"Saving model to {args.output}/final")
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    
    # Save training config and metrics
    config = {
        "model_id": MODEL_ID,
        "base_model": MODEL_ID,
        "task": "epistemic_stance_classification",
        "labels": list(LABEL_TO_ID.keys()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": effective_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_length": MAX_LENGTH,
        "system_prompt": SYSTEM_PROMPT,
        "dataset_stats": dataset_stats,
        "training_metrics": train_result.metrics,
        "final_metrics": gen_metrics,
    }
    
    with open(f"{args.output}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Also save as model card
    model_card = f"""---
tags:
- epistemic-stance
- classification
- magistral
- fine-tuned
base_model: {MODEL_ID}
datasets:
- custom
metrics:
- accuracy
- f1
---

# Epistemic Stance Classifier

Fine-tuned from `{MODEL_ID}` for epistemic stance classification based on Kuhn's developmental epistemology framework.

## Labels

- **absolutist**: Knowledge presented as certain facts
- **multiplist**: Knowledge treated as subjective opinion  
- **evaluativist**: Knowledge as reasoned judgment

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | {gen_metrics.get('accuracy', 0):.4f} |
| F1 (macro) | {gen_metrics.get('f1_macro', 0):.4f} |
| F1 (weighted) | {gen_metrics.get('f1_weighted', 0):.4f} |

### Per-class F1

| Class | F1 |
|-------|-----|
| absolutist | {gen_metrics.get('f1_absolutist', 0):.4f} |
| multiplist | {gen_metrics.get('f1_multiplist', 0):.4f} |
| evaluativist | {gen_metrics.get('f1_evaluativist', 0):.4f} |

## Training

- Epochs: {args.epochs}
- Effective batch size: {effective_batch_size}
- Learning rate: {args.learning_rate}
- Training samples: {dataset_stats.get('train_samples', 'N/A')}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.hub_model_id or 'your-username/epistemic-stance-classifier'}")
tokenizer = AutoTokenizer.from_pretrained("{args.hub_model_id or 'your-username/epistemic-stance-classifier'}")

# See inference script for full usage
```
"""
    
    with open(f"{args.output}/final/README.md", 'w') as f:
        f.write(model_card)
    
    # ==========================================================================
    # PUSH TO HUB (FINAL)
    # ==========================================================================
    
    if push_to_hub:
        logger.info(f"Pushing final model to HuggingFace Hub: {args.hub_model_id}")
        try:
            trainer.push_to_hub(
                commit_message=f"Final model after {args.epochs} epochs - accuracy: {gen_metrics.get('accuracy', 0):.4f}",
            )
            
            # Also push the config and README
            from huggingface_hub import HfApi
            api = HfApi()
            
            api.upload_file(
                path_or_fileobj=f"{args.output}/training_config.json",
                path_in_repo="training_config.json",
                repo_id=args.hub_model_id,
                commit_message="Add training config"
            )
            
            logger.info(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
    
    # ==========================================================================
    # FINISH W&B
    # ==========================================================================
    
    if not args.no_wandb:
        try:
            import wandb
            
            # Log final artifact
            artifact = wandb.Artifact(
                name=f"epistemic-stance-model",
                type="model",
                description=f"Fine-tuned Magistral for epistemic stance classification"
            )
            artifact.add_dir(f"{args.output}/final")
            wandb.log_artifact(artifact)
            
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to finalize W&B: {e}")
    
    logger.info("Training complete!")
    logger.info(f"Final accuracy: {gen_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Final F1 (macro): {gen_metrics.get('f1_macro', 0):.4f}")


if __name__ == '__main__':
    main()
