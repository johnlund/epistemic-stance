#!/usr/bin/env python3
"""
Finalize training: run generation-based evaluation, save config, push to Hub, finish W&B.

Run this after training completes to:
1. Run generation-based evaluation on the fine-tuned model
2. Save training config with metrics
3. Push model to HuggingFace Hub
4. Finalize W&B run

Usage:
    python finalize_training.py --model ./epistemic_stance_model --hub-model-id johnclund/epistemic-stance-analyzer
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, upload_folder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"
LABEL_TO_ID = {"absolutist": 0, "multiplist": 1, "evaluativist": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

SYSTEM_PROMPT = """You are an expert classifier trained to identify epistemic stances in text based on Kuhn's developmental epistemology framework.

## The Three Epistemic Stances

**ABSOLUTIST**: Knowledge is CERTAIN. Claims presented as objective truth without qualification. Dismisses opposing views. Uses directive language like "You need to...", "Obviously...", "The only option is..."

**MULTIPLIST**: Knowledge is SUBJECTIVE. All opinions equally valid. Avoids evaluation. Uses phrases like "Only you can know", "Everyone's entitled to their view", "It depends on the person"

**EVALUATIVIST**: Knowledge is UNCERTAIN but some claims have MORE MERIT. Weighs evidence, engages counterarguments, shows calibrated confidence. Uses "The evidence suggests...", "On balance...", "Based on what you've described..."

## Key Distinctions
- Absolutist vs Evaluativist: Do they JUSTIFY with reasoning, or assert as fact?
- Multiplist vs Evaluativist: Do they WEIGH perspectives, or treat all as equal?

Respond with JSON: {"stance": "absolutist|multiplist|evaluativist", "confidence": "high|medium|low"}"""


# =============================================================================
# EVALUATION
# =============================================================================

def load_eval_data(data_path: str, test_size: float = 0.1, seed: int = 42):
    """Load evaluation data from CSV."""
    logger.info(f"Loading evaluation data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Use same split as training
    from sklearn.model_selection import train_test_split
    _, eval_df = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df['label']
    )
    
    logger.info(f"Evaluation samples: {len(eval_df)}")
    logger.info(f"Label distribution:\n{eval_df['label'].value_counts()}")
    
    return eval_df


def evaluate_model(model, tokenizer, eval_df, num_samples: int = 100, temperature: float = 0.15):
    """Run generation-based evaluation."""
    logger.info(f"Running generation-based evaluation on {num_samples} samples...")
    
    # Sample if needed
    if len(eval_df) > num_samples:
        eval_df = eval_df.sample(n=num_samples, random_state=42)
    
    predictions = []
    references = []
    raw_outputs = []
    
    model.eval()
    with torch.no_grad():
        for idx, row in eval_df.iterrows():
            text = row['text']
            label = row['label']
            references.append(label)
            
            # Build prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify the epistemic stance:\n\n{text}"}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            raw_outputs.append(response)
            
            # Parse the response
            try:
                result = json.loads(response)
                pred_label = result.get('stance', 'unknown')
            except json.JSONDecodeError:
                pred_label = 'unknown'
                for label_name in LABEL_TO_ID:
                    if label_name in response.lower():
                        pred_label = label_name
                        break
            
            predictions.append(pred_label)
            
            # Progress
            if (len(predictions)) % 20 == 0:
                logger.info(f"  Processed {len(predictions)}/{len(eval_df)}")
    
    # Calculate metrics
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p in LABEL_TO_ID]
    invalid_count = len(predictions) - len(valid_pairs)
    
    if not valid_pairs:
        logger.error("No valid predictions!")
        return {"accuracy": 0.0, "valid_predictions": 0}
    
    preds, refs = zip(*valid_pairs)
    
    metrics = {
        "accuracy": accuracy_score(refs, preds),
        "f1_macro": f1_score(refs, preds, average='macro', zero_division=0),
        "f1_weighted": f1_score(refs, preds, average='weighted', zero_division=0),
        "valid_predictions": len(valid_pairs),
        "invalid_predictions": invalid_count,
        "total_samples": len(predictions),
    }
    
    # Per-class metrics
    for label in LABEL_TO_ID:
        label_preds = [1 if p == label else 0 for p in preds]
        label_refs = [1 if r == label else 0 for r in refs]
        metrics[f"f1_{label}"] = f1_score(label_refs, label_preds, zero_division=0)
    
    # Log detailed results
    logger.info(f"\n{classification_report(refs, preds, zero_division=0)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(refs, preds)}")
    
    return metrics, predictions, references, raw_outputs


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Finalize training run')
    parser.add_argument('--model', '-m', required=True, help='Path to fine-tuned model')
    parser.add_argument('--data', '-d', default='final_training_data_balanced.csv', help='Training data CSV')
    parser.add_argument('--hub-model-id', default='johnclund/epistemic-stance-analyzer', help='HuggingFace Hub model ID')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples for evaluation')
    parser.add_argument('--no-push', action='store_true', help='Skip pushing to Hub')
    parser.add_argument('--no-wandb', action='store_true', help='Skip W&B logging')
    parser.add_argument('--wandb-project', default='epistemic-stance', help='W&B project name')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    # ==========================================================================
    # 1. Load model and tokenizer
    # ==========================================================================
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    logger.info("Model loaded successfully")
    
    # ==========================================================================
    # 2. Run generation-based evaluation
    # ==========================================================================
    eval_df = load_eval_data(args.data)
    metrics, predictions, references, raw_outputs = evaluate_model(
        model, tokenizer, eval_df, num_samples=args.num_samples
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    logger.info(f"Valid predictions: {metrics['valid_predictions']}/{metrics['total_samples']}")
    logger.info(f"{'='*60}\n")
    
    # ==========================================================================
    # 3. Save training config with metrics
    # ==========================================================================
    config = {
        "base_model_id": BASE_MODEL_ID,
        "fine_tuned_model": str(model_path),
        "task": "epistemic_stance_classification",
        "labels": list(LABEL_TO_ID.keys()),
        "final_metrics": metrics,
        "eval_samples": args.num_samples,
    }
    
    config_path = model_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {config_path}")
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'text': eval_df['text'].values[:len(predictions)],
        'true_label': references,
        'predicted_label': predictions,
        'raw_output': raw_outputs,
        'correct': [p == r for p, r in zip(predictions, references)]
    })
    predictions_path = model_path / "eval_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # ==========================================================================
    # 4. Push to HuggingFace Hub
    # ==========================================================================
    if not args.no_push and args.hub_model_id:
        logger.info(f"Pushing model to HuggingFace Hub: {args.hub_model_id}")
        try:
            api = HfApi()
            api.create_repo(repo_id=args.hub_model_id, exist_ok=True)
            
            # Create a model card
            model_card = f"""---
license: apache-2.0
base_model: {BASE_MODEL_ID}
tags:
- text-classification
- epistemic-stance
- mistral
- fine-tuned
metrics:
- accuracy
- f1
---

# Epistemic Stance Classifier

Fine-tuned [{BASE_MODEL_ID}](https://huggingface.co/{BASE_MODEL_ID}) for epistemic stance classification.

## Model Description

This model classifies text into three epistemic stances based on Kuhn's developmental epistemology framework:

- **Absolutist**: Knowledge is certain, claims presented as objective truth
- **Multiplist**: Knowledge is subjective, all opinions equally valid  
- **Evaluativist**: Knowledge is uncertain but some claims have more merit

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| F1 Macro | {metrics['f1_macro']:.4f} |
| F1 Weighted | {metrics['f1_weighted']:.4f} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{args.hub_model_id}")

# See inference script for full usage
```
"""
            
            readme_path = model_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(model_card)
            
            upload_folder(
                folder_path=str(model_path),
                repo_id=args.hub_model_id,
                ignore_patterns=["checkpoint-*", "*.pth", "optimizer.pt", "scheduler.pt", "global_step*", "wandb/*"],
                commit_message=f"Final model - accuracy: {metrics['accuracy']:.4f}",
            )
            
            logger.info(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")
            
        except Exception as e:
            logger.error(f"Push to Hub failed: {e}")
    
    # ==========================================================================
    # 5. Finish W&B
    # ==========================================================================
    if not args.no_wandb:
        try:
            import wandb
            
            run = wandb.init(
                project=args.wandb_project,
                job_type="evaluation",
                name="final-evaluation",
                config={
                    "model_path": str(model_path),
                    "num_samples": args.num_samples,
                }
            )
            
            wandb.log({f"final/{k}": v for k, v in metrics.items()})
            
            # Log confusion matrix
            wandb.log({
                "final/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=references,
                    preds=predictions,
                    class_names=list(LABEL_TO_ID.keys())
                )
            })
            
            wandb.finish()
            logger.info("W&B run finalized")
            
        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")
    
    # ==========================================================================
    # Done!
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING FINALIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Final accuracy: {metrics['accuracy']:.4f}")
    if not args.no_push:
        logger.info(f"HuggingFace Hub: https://huggingface.co/{args.hub_model_id}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
