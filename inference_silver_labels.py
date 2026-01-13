#!/usr/bin/env python3
"""
Epistemic Stance Classifier Inference Script
=============================================

Uses a trained Longformer classifier to generate silver labels
for a larger dataset, with confidence filtering for quality control.

Author: Claude (Anthropic)
Project: Epistemic Stance Analysis Pipeline
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LongformerForSequenceClassification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

LABEL2ID = {"absolutist": 0, "evaluativist": 1, "multiplist": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ==============================================================================
# DATASET
# ==============================================================================

class InferenceDataset(Dataset):
    """Simple dataset for inference."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'idx': idx
        }


# ==============================================================================
# TEMPERATURE SCALING
# ==============================================================================

class TemperatureScaling(torch.nn.Module):
    """Temperature scaling for calibrated probabilities."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor([temperature]))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


# ==============================================================================
# INFERENCE FUNCTIONS
# ==============================================================================

def load_model(
    model_path: str,
    device: torch.device
) -> Tuple[LongformerForSequenceClassification, AutoTokenizer, Optional[TemperatureScaling]]:
    """Load trained model, tokenizer, and temperature scaler."""
    
    logger.info(f"Loading model from {model_path}")
    
    model = LongformerForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load temperature scaler if available
    temp_scaler = None
    temp_path = os.path.join(model_path, 'temperature_scaler.pt')
    if os.path.exists(temp_path):
        temp_scaler = TemperatureScaling()
        temp_scaler.load_state_dict(torch.load(temp_path))
        logger.info(f"Loaded temperature scaler (T={temp_scaler.temperature.item():.4f})")
    
    return model, tokenizer, temp_scaler


def predict_batch(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    temp_scaler: Optional[TemperatureScaling] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a batch."""
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if temp_scaler is not None:
            logits = temp_scaler(logits)
        
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
    
    return predictions, probs


def generate_silver_labels(
    model_path: str,
    data_path: str,
    output_path: str,
    text_column: str = 'text',
    confidence_threshold: float = 0.8,
    batch_size: int = 4,
    max_length: int = 2048,
    include_all: bool = False
) -> pd.DataFrame:
    """
    Generate silver labels for a dataset.
    
    Args:
        model_path: Path to trained model
        data_path: Path to CSV with texts to label
        output_path: Where to save labeled data
        text_column: Name of column containing text
        confidence_threshold: Minimum confidence to include sample (default 0.8)
        batch_size: Inference batch size
        max_length: Maximum sequence length
        include_all: If True, include all samples regardless of confidence
    
    Returns:
        DataFrame with silver labels
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, tokenizer, temp_scaler = load_model(model_path, device)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Total samples: {len(df)}")
    
    # Check for text column
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
    
    # Create dataset and loader
    dataset = InferenceDataset(
        texts=df[text_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Run inference
    all_predictions = []
    all_probs = []
    all_indices = []
    
    logger.info("Running inference...")
    for batch in tqdm(loader):
        predictions, probs = predict_batch(model, batch, device, temp_scaler)
        
        all_predictions.extend(predictions)
        all_probs.extend(probs)
        all_indices.extend(batch['idx'].numpy())
    
    # Organize results
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    all_indices = np.array(all_indices)
    
    # Sort by original index (should already be sorted, but just in case)
    sort_order = np.argsort(all_indices)
    all_predictions = all_predictions[sort_order]
    all_probs = all_probs[sort_order]
    
    # Add predictions to dataframe
    df['silver_label'] = [ID2LABEL[p] for p in all_predictions]
    df['silver_confidence'] = np.max(all_probs, axis=1)
    df['prob_absolutist'] = all_probs[:, 0]
    df['prob_evaluativist'] = all_probs[:, 1]
    df['prob_multiplist'] = all_probs[:, 2]
    
    # Compute confidence categories
    df['confidence_category'] = pd.cut(
        df['silver_confidence'],
        bins=[0, 0.6, 0.8, 0.9, 1.0],
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Statistics
    logger.info("\n=== SILVER LABELING RESULTS ===")
    logger.info(f"\nLabel distribution (all samples):")
    logger.info(df['silver_label'].value_counts())
    
    logger.info(f"\nConfidence distribution:")
    logger.info(df['confidence_category'].value_counts())
    
    logger.info(f"\nConfidence statistics:")
    logger.info(f"  Mean: {df['silver_confidence'].mean():.4f}")
    logger.info(f"  Median: {df['silver_confidence'].median():.4f}")
    logger.info(f"  Min: {df['silver_confidence'].min():.4f}")
    logger.info(f"  Max: {df['silver_confidence'].max():.4f}")
    
    # Filter by confidence
    if not include_all:
        high_conf_mask = df['silver_confidence'] >= confidence_threshold
        df_filtered = df[high_conf_mask].copy()
        
        logger.info(f"\n=== FILTERED RESULTS (confidence >= {confidence_threshold}) ===")
        logger.info(f"Samples passing threshold: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")
        logger.info(f"\nFiltered label distribution:")
        logger.info(df_filtered['silver_label'].value_counts())
        
        # Save filtered version
        filtered_output = output_path.replace('.csv', f'_filtered_{confidence_threshold}.csv')
        df_filtered.to_csv(filtered_output, index=False)
        logger.info(f"\nFiltered data saved to: {filtered_output}")
    
    # Save full results
    df.to_csv(output_path, index=False)
    logger.info(f"Full results saved to: {output_path}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(df),
        'confidence_threshold': confidence_threshold,
        'samples_above_threshold': int((df['silver_confidence'] >= confidence_threshold).sum()),
        'label_distribution': df['silver_label'].value_counts().to_dict(),
        'confidence_stats': {
            'mean': float(df['silver_confidence'].mean()),
            'median': float(df['silver_confidence'].median()),
            'std': float(df['silver_confidence'].std()),
            'min': float(df['silver_confidence'].min()),
            'max': float(df['silver_confidence'].max()),
        },
        'per_class_confidence': {
            label: float(df[df['silver_label'] == label]['silver_confidence'].mean())
            for label in ID2LABEL.values()
        }
    }
    
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")
    
    return df


# ==============================================================================
# COMBINING GOLD AND SILVER LABELS
# ==============================================================================

def combine_gold_silver(
    gold_path: str,
    silver_path: str,
    output_path: str,
    gold_weight: float = 2.0,
    silver_confidence_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Combine gold (human/LLM labeled) and silver (classifier labeled) datasets.
    
    Args:
        gold_path: Path to gold labeled CSV
        silver_path: Path to silver labeled CSV
        output_path: Where to save combined data
        gold_weight: Weight multiplier for gold samples during training
        silver_confidence_threshold: Minimum confidence for silver samples
    
    Returns:
        Combined DataFrame ready for fine-tuning
    """
    
    logger.info("Combining gold and silver labels...")
    
    # Load gold data
    gold_df = pd.read_csv(gold_path)
    gold_df['data_source'] = 'gold'
    gold_df['sample_weight'] = gold_weight
    
    # Standardize column names
    if 'epistemic_stance' in gold_df.columns:
        gold_df['label'] = gold_df['epistemic_stance']
    
    logger.info(f"Gold samples: {len(gold_df)}")
    
    # Load silver data
    silver_df = pd.read_csv(silver_path)
    
    # Filter by confidence
    silver_df = silver_df[silver_df['silver_confidence'] >= silver_confidence_threshold].copy()
    silver_df['data_source'] = 'silver'
    silver_df['label'] = silver_df['silver_label']
    silver_df['sample_weight'] = silver_df['silver_confidence']  # Weight by confidence
    
    logger.info(f"Silver samples (above {silver_confidence_threshold}): {len(silver_df)}")
    
    # Select common columns
    common_cols = ['text', 'label', 'data_source', 'sample_weight']
    
    # Add sample_id if available
    if 'sample_id' in gold_df.columns:
        common_cols.insert(0, 'sample_id')
    elif 'sample_id' not in silver_df.columns:
        gold_df['sample_id'] = [f'gold_{i}' for i in range(len(gold_df))]
        silver_df['sample_id'] = [f'silver_{i}' for i in range(len(silver_df))]
        common_cols.insert(0, 'sample_id')
    
    # Combine
    gold_subset = gold_df[common_cols].copy()
    silver_subset = silver_df[common_cols].copy()
    
    combined = pd.concat([gold_subset, silver_subset], ignore_index=True)
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"\nCombined dataset: {len(combined)} samples")
    logger.info(f"  Gold: {(combined['data_source'] == 'gold').sum()}")
    logger.info(f"  Silver: {(combined['data_source'] == 'silver').sum()}")
    logger.info(f"\nLabel distribution:")
    logger.info(combined['label'].value_counts())
    
    # Save
    combined.to_csv(output_path, index=False)
    logger.info(f"\nCombined data saved to: {output_path}")
    
    return combined


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate silver labels with trained classifier")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate silver labels')
    predict_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--data', type=str, required=True, help='Path to unlabeled CSV')
    predict_parser.add_argument('--output', type=str, required=True, help='Output path')
    predict_parser.add_argument('--text-column', type=str, default='text', help='Text column name')
    predict_parser.add_argument('--threshold', type=float, default=0.8, 
                                help='Confidence threshold')
    predict_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    predict_parser.add_argument('--max-length', type=int, default=2048, help='Max sequence length')
    predict_parser.add_argument('--include-all', action='store_true',
                                help='Include all samples regardless of confidence')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine gold and silver labels')
    combine_parser.add_argument('--gold', type=str, required=True, help='Path to gold labels')
    combine_parser.add_argument('--silver', type=str, required=True, help='Path to silver labels')
    combine_parser.add_argument('--output', type=str, required=True, help='Output path')
    combine_parser.add_argument('--gold-weight', type=float, default=2.0, 
                                help='Weight for gold samples')
    combine_parser.add_argument('--threshold', type=float, default=0.8,
                                help='Silver confidence threshold')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        generate_silver_labels(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            text_column=args.text_column,
            confidence_threshold=args.threshold,
            batch_size=args.batch_size,
            max_length=args.max_length,
            include_all=args.include_all
        )
    
    elif args.command == 'combine':
        combine_gold_silver(
            gold_path=args.gold,
            silver_path=args.silver,
            output_path=args.output,
            gold_weight=args.gold_weight,
            silver_confidence_threshold=args.threshold
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
