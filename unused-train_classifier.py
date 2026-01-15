"""
Epistemic Stance Classifier Training
=====================================

Train a DeBERTa-based classifier on Claude-labeled epistemic stance data.

Requirements:
    pip install transformers datasets torch scikit-learn pandas accelerate

Usage:
    python train_classifier.py --input wildchat_labeled.csv --output ./epistemic_classifier
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

LABEL_MAP = {
    'absolutist': 0,
    'multiplist': 1,
    'evaluativist': 2
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

MODEL_NAME = "microsoft/deberta-v3-base"  # Good balance of performance and size
# Alternatives:
# - "microsoft/deberta-v3-large" - better performance, more compute
# - "roberta-base" - faster, slightly less accurate
# - "distilbert-base-uncased" - fastest, good for prototyping


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data(csv_path, test_size=0.15, val_size=0.15, min_confidence='low'):
    """
    Load labeled data and prepare train/val/test splits.
    
    Args:
        csv_path: Path to labeled CSV
        test_size: Fraction for test set
        val_size: Fraction for validation set
        min_confidence: Minimum confidence to include ('low', 'medium', 'high')
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter by confidence if specified
    confidence_order = {'low': 0, 'medium': 1, 'high': 2}
    min_conf_val = confidence_order.get(min_confidence, 0)
    
    df['conf_val'] = df['stance_confidence'].map(confidence_order)
    df = df[df['conf_val'] >= min_conf_val]
    
    # Filter out errors
    df = df[df['epistemic_stance'].isin(LABEL_MAP.keys())]
    
    print(f"After filtering: {len(df)} samples")
    print(f"\nLabel distribution:")
    print(df['epistemic_stance'].value_counts())
    
    # Add numeric labels
    df['label'] = df['epistemic_stance'].map(LABEL_MAP)
    
    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), 
        stratify=df['label'], random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size/(test_size + val_size),
        stratify=temp_df['label'], random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def create_hf_dataset(df, tokenizer, max_length=512):
    """Convert DataFrame to HuggingFace Dataset with tokenization."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['content'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    # Create HF dataset
    dataset = HFDataset.from_pandas(df[['content', 'label']].reset_index(drop=True))
    
    # Tokenize
    dataset = dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return dataset


# ============================================================================
# MODEL TRAINING
# ============================================================================

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate per-class metrics
    report = classification_report(
        labels, predictions, 
        target_names=list(LABEL_MAP.keys()),
        output_dict=True
    )
    
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'absolutist_f1': report['absolutist']['f1-score'],
        'multiplist_f1': report['multiplist']['f1-score'],
        'evaluativist_f1': report['evaluativist']['f1-score'],
    }


def train_model(train_dataset, val_dataset, output_dir, 
                model_name=MODEL_NAME, epochs=5, batch_size=16,
                learning_rate=2e-5, weight_decay=0.01):
    """
    Train the classifier.
    """
    print(f"\nLoading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save best model
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    print(f"\nModel saved to {output_dir}/best_model")
    
    return trainer, tokenizer


def evaluate_model(trainer, test_dataset, output_dir):
    """Evaluate on test set and save results."""
    
    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)
    
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        true_labels, pred_labels,
        target_names=list(LABEL_MAP.keys())
    )
    print(report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(pd.DataFrame(
        cm, 
        index=list(LABEL_MAP.keys()),
        columns=list(LABEL_MAP.keys())
    ))
    
    # Save results
    results = {
        'classification_report': classification_report(
            true_labels, pred_labels,
            target_names=list(LABEL_MAP.keys()),
            output_dict=True
        ),
        'confusion_matrix': cm.tolist()
    }
    
    with open(f"{output_dir}/test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/test_results.json")
    
    return results


# ============================================================================
# INFERENCE
# ============================================================================

class EpistemicStanceClassifier:
    """Wrapper for easy inference with trained model."""
    
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def predict(self, text, return_probs=False):
        """
        Predict epistemic stance for a single text.
        
        Args:
            text: Input text string
            return_probs: Whether to return probability distribution
        
        Returns:
            Dictionary with prediction and optionally probabilities
        """
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        
        result = {
            'stance': ID_TO_LABEL[pred_label],
            'confidence': float(probs[pred_label])
        }
        
        if return_probs:
            result['probabilities'] = {
                ID_TO_LABEL[i]: float(p) for i, p in enumerate(probs)
            }
        
        return result
    
    def predict_batch(self, texts, batch_size=32):
        """Predict epistemic stance for multiple texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            pred_labels = np.argmax(probs, axis=-1)
            
            for j, (label, prob) in enumerate(zip(pred_labels, probs)):
                results.append({
                    'stance': ID_TO_LABEL[label],
                    'confidence': float(prob[label])
                })
        
        return results


def label_full_dataset(classifier, input_csv, output_csv, batch_size=32):
    """
    Use trained classifier to label full dataset.
    """
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Labeling {len(df)} samples...")
    texts = df['content'].tolist()
    
    results = classifier.predict_batch(texts, batch_size=batch_size)
    
    df['epistemic_stance_pred'] = [r['stance'] for r in results]
    df['stance_confidence_pred'] = [r['confidence'] for r in results]
    
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    
    print("\nPredicted distribution:")
    print(df['epistemic_stance_pred'].value_counts())


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train epistemic stance classifier')
    parser.add_argument('--input', required=True, help='Path to labeled CSV')
    parser.add_argument('--output', default='./epistemic_classifier', help='Output directory')
    parser.add_argument('--model', default=MODEL_NAME, help='Base model name')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--min_confidence', default='low', 
                        choices=['low', 'medium', 'high'],
                        help='Minimum label confidence to include')
    
    args = parser.parse_args()
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(
        args.input,
        min_confidence=args.min_confidence
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets
    print("\nTokenizing datasets...")
    train_dataset = create_hf_dataset(train_df, tokenizer)
    val_dataset = create_hf_dataset(val_df, tokenizer)
    test_dataset = create_hf_dataset(test_df, tokenizer)
    
    # Train model
    trainer, tokenizer = train_model(
        train_dataset, val_dataset, args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate
    evaluate_model(trainer, test_dataset, args.output)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"""
    Model saved to: {args.output}/best_model
    
    To use the model for inference:
    
        from train_classifier import EpistemicStanceClassifier
        
        classifier = EpistemicStanceClassifier("{args.output}/best_model")
        result = classifier.predict("Your text here")
        print(result)
        # {{'stance': 'evaluativist', 'confidence': 0.87}}
    
    To label a full dataset:
    
        classifier = EpistemicStanceClassifier("{args.output}/best_model")
        label_full_dataset(classifier, "input.csv", "output.csv")
    """)


if __name__ == "__main__":
    main()
