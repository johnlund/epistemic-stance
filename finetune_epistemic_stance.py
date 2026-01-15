#!/usr/bin/env python3
"""
Full fine-tune Magistral-Small-2509 (24B) for epistemic stance classification.

Optimized for 2x B200 (360 GB VRAM total).

Usage:
    # Single command to run training
    accelerate launch --num_processes 2 --mixed_precision bf16 finetune_epistemic_stance.py \
        --data final_training_data_balanced.csv \
        --output ./epistemic_stance_model \
        --epochs 3

Requirements:
    pip install torch transformers datasets accelerate pandas scikit-learn wandb
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

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
        ]
    }


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.1,
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Load CSV and prepare train/eval datasets."""
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Log distribution
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
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
        examples.append(example)
    
    # Split train/eval
    train_examples, eval_examples = train_test_split(
        examples,
        test_size=test_size,
        random_state=seed,
        stratify=[e['messages'][2]['content'] for e in examples]  # Stratify by label
    )
    
    logger.info(f"Train samples: {len(train_examples)}")
    logger.info(f"Eval samples: {len(eval_examples)}")
    
    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples)


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
    # Find where assistant response starts
    assistant_start_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
    input_ids = tokenized['input_ids']
    
    # Find last occurrence of assistant marker
    labels = tokenized['labels']
    mask_until = 0
    for i in range(len(input_ids) - len(assistant_start_tokens)):
        if input_ids[i:i+len(assistant_start_tokens)] == assistant_start_tokens:
            mask_until = i + len(assistant_start_tokens)
    
    # Mask everything before assistant response with -100
    for i in range(mask_until):
        labels[i] = -100
    
    tokenized['labels'] = labels
    
    return tokenized


# =============================================================================
# CUSTOM TRAINER WITH SAMPLE WEIGHTS
# =============================================================================

class WeightedTrainer(Trainer):
    """Trainer that supports sample weights in loss computation."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss."""
        
        # Extract sample weights if present
        sample_weights = inputs.pop('sample_weight', None)
        
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
    parser.add_argument('--wandb-project', type=str, default='epistemic-stance', help='W&B project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    args = parser.parse_args()
    
    # Setup W&B
    if not args.no_wandb:
        os.environ['WANDB_PROJECT'] = args.wandb_project
    else:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Load tokenizer
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
    
    # Load and prepare data
    train_dataset, eval_dataset = load_and_prepare_data(
        args.data,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    
    def tokenize_fn(example):
        tokenized = tokenize_conversation(example, tokenizer)
        tokenized['sample_weight'] = example['sample_weight']
        return tokenized
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        remove_columns=['messages'],
        num_proc=8,
        desc="Tokenizing train"
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_fn,
        remove_columns=['messages'],
        num_proc=8,
        desc="Tokenizing eval"
    )
    
    # Load model
    logger.info(f"Loading model from {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Use Flash Attention for efficiency
        device_map="auto",  # Distribute across GPUs
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation * 2  # 2 GPUs
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Training arguments
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
        
        # Misc
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"epistemic-stance-magistral-{args.epochs}ep",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output}/final")
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    
    # Save training config
    config = {
        "model_id": MODEL_ID,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": effective_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_length": MAX_LENGTH,
        "system_prompt": SYSTEM_PROMPT,
    }
    
    with open(f"{args.output}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
