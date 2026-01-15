#!/usr/bin/env python3
"""
Fine-tune Magistral-Small-2509 (24B) for epistemic stance classification.

Uses Mistral3ForConditionalGeneration as specified in the HuggingFace model card.

Usage:
    accelerate launch --config_file accelerate_config.yaml finetune_epistemic_stance.py \
        --data final_training_data_balanced.csv \
        --output ./epistemic_stance_model \
        --epochs 3
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from transformers import (
    Mistral3ForConditionalGeneration,  # Correct model class for Magistral
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
MAX_LENGTH = 2048

# Label mapping
LABEL_TO_ID = {"absolutist": 0, "multiplist": 1, "evaluativist": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Our classification system prompt (simplified, without thinking tokens)
CLASSIFICATION_SYSTEM_PROMPT = """You are an expert classifier trained to identify epistemic stances in text based on Kuhn's developmental epistemology framework.

## The Three Epistemic Stances

**ABSOLUTIST**: Knowledge is CERTAIN. Claims presented as objective truth without qualification. Dismisses opposing views. Uses directive language like "You need to...", "Obviously...", "The only option is..."

**MULTIPLIST**: Knowledge is SUBJECTIVE. All opinions equally valid. Avoids evaluation. Uses phrases like "Only you can know", "Everyone's entitled to their view", "It depends on the person"

**EVALUATIVIST**: Knowledge is UNCERTAIN but some claims have MORE MERIT. Weighs evidence, engages counterarguments, shows calibrated confidence. Uses "The evidence suggests...", "On balance...", "Based on what you've described..."

## Key Distinctions
- Absolutist vs Evaluativist: Do they JUSTIFY with reasoning, or assert as fact?
- Multiplist vs Evaluativist: Do they WEIGH perspectives, or treat all as equal?

Respond with JSON: {"stance": "absolutist|multiplist|evaluativist", "confidence": "high|medium|low"}"""


# =============================================================================
# DATA PREPARATION
# =============================================================================

def format_training_example(text: str, label: str, confidence: str = "high") -> dict:
    """Format a single training example."""
    return {
        "messages": [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the epistemic stance:\n\n{text}"},
            {"role": "assistant", "content": json.dumps({"stance": label, "confidence": confidence})}
        ],
        "label": label,
    }


def load_and_prepare_data(data_path: str, test_size: float = 0.1, seed: int = 42):
    """Load CSV and prepare datasets."""
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    required_cols = ['text', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    dataset_stats = {
        "total_samples": len(df),
        "label_distribution": df['label'].value_counts().to_dict(),
    }
    
    def get_confidence(row):
        weight = row.get('sample_weight', 1.0)
        return "high" if weight >= 2.0 else "medium" if weight >= 1.0 else "low"
    
    examples = []
    for _, row in df.iterrows():
        example = format_training_example(
            text=row['text'],
            label=row['label'],
            confidence=get_confidence(row)
        )
        example['sample_weight'] = row.get('sample_weight', 1.0)
        example['text_original'] = row['text']
        examples.append(example)
    
    train_examples, eval_examples = train_test_split(
        examples, test_size=test_size, random_state=seed,
        stratify=[e['label'] for e in examples]
    )
    
    logger.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    dataset_stats["train_samples"] = len(train_examples)
    dataset_stats["eval_samples"] = len(eval_examples)
    
    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples), dataset_stats


def tokenize_example(example: dict, tokenizer) -> dict:
    """Tokenize a single example."""
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    # Mask everything except the assistant response
    # Find the assistant response start
    input_ids = tokenized['input_ids']
    labels = tokenized['labels']
    
    # Try to find where assistant response starts
    # The tokenizer should handle this, but we'll mask the prompt portion
    assistant_response = json.dumps({"stance": example['label'], "confidence": "high"})
    response_tokens = tokenizer.encode(assistant_response, add_special_tokens=False)
    
    # Find response position and mask everything before it
    mask_until = len(input_ids) - len(response_tokens) - 5  # Some buffer
    mask_until = max(0, mask_until)
    
    for i in range(mask_until):
        labels[i] = -100
    
    tokenized['labels'] = labels
    tokenized['label_id'] = LABEL_TO_ID.get(example['label'], -1)
    
    return tokenized


# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class WeightedTrainer(Trainer):
    """Trainer with sample weight support."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weights = inputs.pop('sample_weight', None)
        inputs.pop('label_id', None)
        
        outputs = model(**inputs)
        
        if sample_weights is not None:
            logits = outputs.logits
            labels = inputs['labels']
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            mask = (shift_labels != -100).float()
            per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            loss = (per_sample_loss * sample_weights.to(per_sample_loss.device)).mean()
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, tokenizer, eval_dataset, num_samples: int = 100):
    """Run generation-based evaluation."""
    import random
    from sklearn.metrics import accuracy_score, f1_score
    
    indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
    
    predictions, references = [], []
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            example = eval_dataset[idx]
            ref_label = example.get('label', 'unknown')
            references.append(ref_label)
            
            messages = [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify the epistemic stance:\n\n{example.get('text_original', '')}"}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            try:
                result = json.loads(response)
                pred_label = result.get('stance', 'unknown')
            except json.JSONDecodeError:
                pred_label = 'unknown'
                for label in LABEL_TO_ID:
                    if label in response.lower():
                        pred_label = label
                        break
            
            predictions.append(pred_label)
    
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p in LABEL_TO_ID and r in LABEL_TO_ID]
    
    if not valid_pairs:
        return {"accuracy": 0.0, "valid_predictions": 0}
    
    preds, refs = zip(*valid_pairs)
    
    return {
        "accuracy": accuracy_score(refs, preds),
        "f1_macro": f1_score(refs, preds, average='macro', zero_division=0),
        "f1_weighted": f1_score(refs, preds, average='weighted', zero_division=0),
        "valid_predictions": len(valid_pairs),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb-project', type=str, default='epistemic-stance')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--hub-model-id', type=str, default=None)
    parser.add_argument('--hub-private', action='store_true')
    parser.add_argument('--no-push', action='store_true')
    
    args = parser.parse_args()
    
    # W&B setup
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            logger.info(f"W&B initialized: {wandb.run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            args.no_wandb = True
    
    if args.no_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Load tokenizer (with mistral tokenizer type as per docs)
    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        tokenizer_type="mistral",
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load data
    train_dataset, eval_dataset, dataset_stats = load_and_prepare_data(
        args.data, test_size=args.test_size, seed=args.seed
    )
    
    eval_dataset_raw = eval_dataset
    
    # Tokenize
    logger.info("Tokenizing datasets...")
    
    def tokenize_fn(example):
        tokenized = tokenize_example(example, tokenizer)
        tokenized['sample_weight'] = example['sample_weight']
        return tokenized
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        remove_columns=['messages', 'label', 'text_original'],
        num_proc=4,
        desc="Tokenizing train"
    )
    
    eval_dataset_tokenized = eval_dataset.map(
        tokenize_fn,
        remove_columns=['messages'],
        num_proc=4,
        desc="Tokenizing eval"
    )
    
    # Load model (using correct class as per HuggingFace docs)
    logger.info(f"Loading model from {MODEL_ID}")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    model.gradient_checkpointing_enable()
    
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = args.batch_size * args.gradient_accumulation * num_gpus
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Training arguments
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
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        push_to_hub=push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        hub_private_repo=args.hub_private,
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else "none",
    )
    
    # Data collator
    class CustomDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            sample_weights = [f.pop('sample_weight', 1.0) for f in features]
            label_ids = [f.pop('label_id', -1) for f in features]
            for f in features:
                f.pop('label', None)
                f.pop('text_original', None)
            batch = super().__call__(features)
            batch['sample_weight'] = torch.tensor(sample_weights, dtype=torch.float32)
            batch['label_id'] = torch.tensor(label_ids, dtype=torch.long)
            return batch
    
    data_collator = CustomDataCollator(tokenizer=tokenizer, padding=True, return_tensors="pt")
    
    # Train
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_tokenized,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training metrics: {train_result.metrics}")
    
    # Evaluate
    logger.info("Running generation-based evaluation...")
    gen_metrics = evaluate_model(model, tokenizer, eval_dataset_raw, num_samples=100)
    logger.info(f"Evaluation results: {gen_metrics}")
    
    # Save
    logger.info(f"Saving model to {args.output}/final")
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    
    config = {
        "model_id": MODEL_ID,
        "task": "epistemic_stance_classification",
        "labels": list(LABEL_TO_ID.keys()),
        "final_metrics": gen_metrics,
        "dataset_stats": dataset_stats,
    }
    
    with open(f"{args.output}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Push to hub
    if push_to_hub:
        logger.info(f"Pushing to Hub: {args.hub_model_id}")
        try:
            trainer.push_to_hub(commit_message=f"Final model - accuracy: {gen_metrics.get('accuracy', 0):.4f}")
        except Exception as e:
            logger.error(f"Push failed: {e}")
    
    # Finish W&B
    if not args.no_wandb:
        try:
            import wandb
            wandb.log({f"final/{k}": v for k, v in gen_metrics.items()})
            wandb.finish()
        except Exception:
            pass
    
    logger.info("Training complete!")
    logger.info(f"Final accuracy: {gen_metrics.get('accuracy', 0):.4f}")


if __name__ == '__main__':
    main()
