#!/usr/bin/env python3
"""
Epistemic Stance Classifier Training Script
============================================

Trains a Longformer-based classifier for epistemic stance detection
(absolutist, multiplist, evaluativist) with:
- Class weighting for imbalanced data
- Optional focal loss for hard example mining
- Confidence-weighted sample loss
- Temperature scaling for calibration
- Comprehensive per-class evaluation metrics

Author: Claude (Anthropic)
Project: Epistemic Stance Analysis Pipeline
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# Optional HuggingFace Hub import
try:
    from huggingface_hub import HfApi, login as hf_login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Log optional dependencies availability
if not WANDB_AVAILABLE:
    logger.info("wandb not available. Install with: pip install wandb to enable experiment tracking")
if not HF_HUB_AVAILABLE:
    logger.info("huggingface_hub not available. Install with: pip install huggingface_hub to enable model uploads")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Training configuration with sensible defaults."""
    
    # Model
    model_name: str = "allenai/longformer-base-4096"
    max_length: int = 2048  # Covers 99%+ of samples
    num_labels: int = 3
    
    # Labels
    label2id: Dict[str, int] = {
        "absolutist": 0,
        "evaluativist": 1,
        "multiplist": 2
    }
    id2label: Dict[int, str] = {v: k for k, v in label2id.items()}
    
    # Training
    batch_size: int = 4  # Small due to long sequences
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Loss
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    
    # Confidence weighting
    confidence_weights: Dict[str, float] = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4
    }
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 200
    metric_for_best_model: str = "f1_macro"
    
    # Calibration
    apply_temperature_scaling: bool = True
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "epistemic-stance-classifier"
    wandb_run_name: Optional[str] = None
    
    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_organization: Optional[str] = None
    
    # Paths
    output_dir: str = "./output"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ==============================================================================
# FOCAL LOSS IMPLEMENTATION
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'none'
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==============================================================================
# DATASET
# ==============================================================================

class EpistemicStanceDataset(Dataset):
    """Dataset for epistemic stance classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int,
        confidence_levels: Optional[List[str]] = None,
        confidence_weights: Optional[Dict[str, float]] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidence_levels = confidence_levels
        self.confidence_weights = confidence_weights or {"high": 1.0, "medium": 0.7, "low": 0.4}
        
        # Compute sample weights based on confidence
        if confidence_levels is not None:
            self.sample_weights = [
                self.confidence_weights.get(conf, 1.0) 
                for conf in confidence_levels
            ]
        else:
            self.sample_weights = [1.0] * len(texts)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        sample_weight = self.sample_weights[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'sample_weight': torch.tensor(sample_weight, dtype=torch.float)
        }


# ==============================================================================
# TEMPERATURE SCALING FOR CALIBRATION
# ==============================================================================

class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling for probability calibration.
    
    Learns a single temperature parameter T such that
    calibrated_probs = softmax(logits / T)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
    
    def calibrate(
        self,
        model,
        val_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50
    ) -> float:
        """Learn optimal temperature on validation set."""
        model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


# ==============================================================================
# METRICS AND EVALUATION
# ==============================================================================

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str]
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }
    
    # Add per-class metrics
    for idx, label_name in id2label.items():
        metrics[f'precision_{label_name}'] = precision[idx]
        metrics[f'recall_{label_name}'] = recall[idx]
        metrics[f'f1_{label_name}'] = f1[idx]
        metrics[f'support_{label_name}'] = int(support[idx])
    
    return metrics


def compute_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual accuracy.
    Lower is better; < 0.1 is generally considered well-calibrated.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower in np.linspace(0, 1, n_bins + 1)[:-1]:
        bin_upper = bin_lower + 1.0 / n_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    config: Config,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        sample_weights = batch['sample_weight'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Compute loss with sample weights
        if isinstance(loss_fn, FocalLoss):
            loss = loss_fn(outputs.logits, labels)
            loss = (loss * sample_weights).mean()
        else:
            loss = F.cross_entropy(outputs.logits, labels, reduction='none')
            loss = (loss * sample_weights).mean()
        
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        progress_bar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def evaluate(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    temperature_scaler: Optional[TemperatureScaling] = None
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply temperature scaling if available
            if temperature_scaler is not None:
                logits = temperature_scaler(logits)
            
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            predictions = np.argmax(probs, axis=1)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_probs.extend(probs)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    metrics = compute_metrics(predictions, labels, id2label)
    metrics['ece'] = compute_calibration_error(probs, labels)
    
    return metrics, predictions, labels, probs


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train(
    data_path: str,
    config: Config,
    output_dir: Optional[str] = None
) -> Tuple[str, Dict[str, float]]:
    """
    Main training function.
    
    Args:
        data_path: Path to CSV with columns: text, epistemic_stance, stance_confidence
        config: Training configuration
        output_dir: Where to save model and results
    
    Returns:
        Path to best model checkpoint and final metrics
    """
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"longformer-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                'model_name': config.model_name,
                'max_length': config.max_length,
                'batch_size': config.batch_size,
                'gradient_accumulation_steps': config.gradient_accumulation_steps,
                'effective_batch_size': config.batch_size * config.gradient_accumulation_steps,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'num_epochs': config.num_epochs,
                'warmup_ratio': config.warmup_ratio,
                'use_focal_loss': config.use_focal_loss,
                'focal_gamma': config.focal_gamma,
                'use_class_weights': config.use_class_weights,
                'apply_temperature_scaling': config.apply_temperature_scaling,
                'data_path': data_path,
            },
            tags=['longformer', 'epistemic-stance', 'classification']
        )
        logger.info("Wandb initialized")
    elif config.use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not available. Continuing without wandb logging.")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert labels to integers
    df['label_id'] = df['epistemic_stance'].map(config.label2id)
    
    logger.info(f"Dataset size: {len(df)}")
    label_dist = df['epistemic_stance'].value_counts().to_dict()
    logger.info(f"Label distribution:\n{label_dist}")
    
    # Log dataset info to wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            'dataset_size': len(df),
            'label_distribution': label_dist
        })
    
    # Compute class weights
    if config.use_class_weights:
        class_counts = df['epistemic_stance'].value_counts()
        total = len(df)
        class_weights = torch.tensor([
            total / (len(config.label2id) * class_counts[config.id2label[i]])
            for i in range(len(config.label2id))
        ], dtype=torch.float).to(device)
        logger.info(f"Class weights: {dict(zip(config.id2label.values(), class_weights.cpu().numpy()))}")
    else:
        class_weights = None
    
    # Train/val/test split (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['label_id'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label_id'], random_state=42
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Log split sizes to wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        })
    
    # Save splits for reproducibility
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id
    )
    model.to(device)
    
    # Create datasets
    train_dataset = EpistemicStanceDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
        confidence_levels=train_df['stance_confidence'].tolist(),
        confidence_weights=config.confidence_weights
    )
    
    val_dataset = EpistemicStanceDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    test_dataset = EpistemicStanceDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging; increase for speed
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Setup loss function
    if config.use_focal_loss:
        loss_fn = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
        logger.info(f"Using Focal Loss with gamma={config.focal_gamma}")
    else:
        loss_fn = None  # Will use standard cross-entropy in training loop
        logger.info("Using standard Cross-Entropy Loss")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Training loop
    best_metric = 0.0
    best_epoch = 0
    best_model_path = None
    training_history = []
    
    for epoch in range(config.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, config, epoch
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, device, config.id2label
        )
        
        logger.info(f"Validation metrics:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save training history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        }
        training_history.append(history_entry)
        
        # Log to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
            }
            # Add all validation metrics with val/ prefix
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    log_dict[f'val/{key}'] = value
                else:
                    log_dict[f'val/{key}'] = value
            wandb.log(log_dict, step=epoch + 1)
        
        # Check if best model
        current_metric = val_metrics[config.metric_for_best_model]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            
            # Save best model
            best_model_path = os.path.join(output_dir, 'best_model')
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"New best model saved! {config.metric_for_best_model}: {best_metric:.4f}")
            
            # Log best metric to wandb
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.run.summary['best_val_metric'] = best_metric
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.log({'best_val_metric': best_metric}, step=epoch + 1)
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint-epoch-{epoch + 1}')
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete! Best epoch: {best_epoch}, Best {config.metric_for_best_model}: {best_metric:.4f}")
    logger.info(f"{'='*50}")
    
    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)
    
    # Temperature scaling calibration
    temperature_scaler = None
    if config.apply_temperature_scaling:
        logger.info("\nApplying temperature scaling calibration...")
        temperature_scaler = TemperatureScaling()
        optimal_temp = temperature_scaler.calibrate(model, val_loader, device)
        logger.info(f"Optimal temperature: {optimal_temp:.4f}")
        
        # Save temperature
        torch.save(temperature_scaler.state_dict(), os.path.join(best_model_path, 'temperature_scaler.pt'))
        
        # Log temperature to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.config.update({'optimal_temperature': optimal_temp})
            wandb.run.summary['optimal_temperature'] = optimal_temp
    
    # Final test evaluation
    logger.info("\nFinal evaluation on test set:")
    test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, config.id2label, temperature_scaler
    )
    
    for key, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Log test metrics to wandb
    if config.use_wandb and WANDB_AVAILABLE:
        for key, value in test_metrics.items():
            if isinstance(value, float):
                wandb.run.summary[f'test/{key}'] = value
            else:
                wandb.run.summary[f'test/{key}'] = value
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"Labels: {list(config.id2label.values())}")
    logger.info(f"\n{cm}")
    
    # Log confusion matrix to wandb
    if config.use_wandb and WANDB_AVAILABLE:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(config.id2label.values()),
                    yticklabels=list(config.id2label.values()))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        wandb.log({'confusion_matrix': wandb.Image(plt)})
        plt.close()
    
    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(
        test_labels, test_preds,
        target_names=list(config.id2label.values())
    )
    logger.info(f"\n{report}")
    
    # Save all results
    results = {
        'config': {
            'model_name': config.model_name,
            'max_length': config.max_length,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'use_focal_loss': config.use_focal_loss,
            'focal_gamma': config.focal_gamma,
            'use_class_weights': config.use_class_weights,
        },
        'best_epoch': best_epoch,
        'best_val_metric': best_metric,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'confusion_matrix': cm.tolist(),
        'temperature': temperature_scaler.temperature.item() if temperature_scaler else None,
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for analysis
    test_df_with_preds = test_df.copy()
    test_df_with_preds['predicted_label'] = [config.id2label[p] for p in test_preds]
    test_df_with_preds['predicted_prob_absolutist'] = test_probs[:, 0]
    test_df_with_preds['predicted_prob_evaluativist'] = test_probs[:, 1]
    test_df_with_preds['predicted_prob_multiplist'] = test_probs[:, 2]
    test_df_with_preds.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    logger.info(f"\nAll results saved to {output_dir}")
    
    # Upload to HuggingFace Hub if requested
    if config.push_to_hub:
        if not HF_HUB_AVAILABLE:
            logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")
        else:
            try:
                hub_model_id = config.hub_model_id
                if not hub_model_id:
                    # Generate model ID from organization or username
                    if config.hub_organization:
                        hub_model_id = f"{config.hub_organization}/epistemic-stance-longformer"
                    else:
                        # Get username from HF token
                        api = HfApi()
                        user = api.whoami()
                        hub_model_id = f"{user['name']}/epistemic-stance-longformer"
                
                logger.info(f"\nUploading model to HuggingFace Hub: {hub_model_id}")
                
                # Upload model and tokenizer
                model.push_to_hub(
                    hub_model_id,
                    private=False,
                    commit_message=f"Add epistemic stance classifier - Best {config.metric_for_best_model}: {best_metric:.4f}"
                )
                tokenizer.push_to_hub(
                    hub_model_id,
                    commit_message="Add tokenizer"
                )
                
                # Upload temperature scaler if available
                if temperature_scaler:
                    temp_path = os.path.join(best_model_path, 'temperature_scaler.pt')
                    if os.path.exists(temp_path):
                        api = HfApi()
                        api.upload_file(
                            path_or_fileobj=temp_path,
                            path_in_repo="temperature_scaler.pt",
                            repo_id=hub_model_id,
                            repo_type="model",
                            commit_message="Add temperature scaler for calibration"
                        )
                
                # Create model card
                model_card = f"""---
license: mit
tags:
- epistemic-stance
- classification
- longformer
- nlp
- social-science
datasets:
- custom
metrics:
- accuracy
- f1
- precision
- recall
---

# Epistemic Stance Classifier

A Longformer-based classifier for detecting epistemic stances (absolutist, evaluativist, multiplist) in text.

## Model Details

- **Model Type**: AutoModelForSequenceClassification (Longformer architecture)
- **Base Model**: {config.model_name}
- **Max Sequence Length**: {config.max_length}
- **Number of Labels**: {config.num_labels}
- **Labels**: absolutist, evaluativist, multiplist

## Training Details

- **Best Validation Metric ({config.metric_for_best_model})**: {best_metric:.4f}
- **Best Epoch**: {best_epoch}
- **Training Epochs**: {config.num_epochs}
- **Learning Rate**: {config.learning_rate}
- **Batch Size**: {config.batch_size}
- **Gradient Accumulation Steps**: {config.gradient_accumulation_steps}
- **Focal Loss**: {config.use_focal_loss}
- **Class Weights**: {config.use_class_weights}
- **Temperature Scaling**: {config.apply_temperature_scaling}

## Test Set Performance

- **Accuracy**: {test_metrics.get('accuracy', 'N/A'):.4f}
- **F1 Macro**: {test_metrics.get('f1_macro', 'N/A'):.4f}
- **F1 Weighted**: {test_metrics.get('f1_weighted', 'N/A'):.4f}
- **Expected Calibration Error**: {test_metrics.get('ece', 'N/A'):.4f}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "{hub_model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "Your text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length={config.max_length})

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()

label_map = {{0: "absolutist", 1: "evaluativist", 2: "multiplist"}}
print(f"Predicted: {{label_map[predicted_class]}}")
print(f"Confidence: {{probs[0][predicted_class]:.4f}}")
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{epistemic-stance-classifier,
  title={{Epistemic Stance Classifier}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{hub_model_id}}}}}
}}
```
"""
                
                # Save and upload model card
                readme_path = os.path.join(best_model_path, 'README.md')
                with open(readme_path, 'w') as f:
                    f.write(model_card)
                
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=hub_model_id,
                    repo_type="model",
                    commit_message="Add model card"
                )
                
                logger.info(f"✅ Model successfully uploaded to: https://huggingface.co/{hub_model_id}")
                
            except Exception as e:
                logger.error(f"Failed to upload model to HuggingFace Hub: {e}")
                logger.info("Model saved locally. You can upload manually later.")
    
    # Finish wandb run
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        logger.info("Wandb run completed")
    
    return best_model_path, test_metrics


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Epistemic Stance Classifier")
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to labeled CSV')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    
    # Model
    parser.add_argument('--model', type=str, default='allenai/longformer-base-4096',
                        help='Model name or path')
    parser.add_argument('--max-length', type=int, default=2048,
                        help='Maximum sequence length')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    
    # Loss
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable class weights')
    
    # Calibration
    parser.add_argument('--no-calibration', action='store_true',
                        help='Disable temperature scaling')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='epistemic-stance-classifier',
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    # HuggingFace Hub
    parser.add_argument('--push-to-hub', action='store_true',
                        help='Upload model to HuggingFace Hub after training')
    parser.add_argument('--hub-model-id', type=str, default=None,
                        help='HuggingFace model ID (e.g., username/model-name or org/model-name)')
    parser.add_argument('--hub-organization', type=str, default=None,
                        help='HuggingFace organization name (if not using hub-model-id)')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        use_class_weights=not args.no_class_weights,
        apply_temperature_scaling=not args.no_calibration,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_organization=args.hub_organization,
        output_dir=args.output_dir
    )
    
    # Train
    train(args.data, config, args.output_dir)


if __name__ == '__main__':
    main()
