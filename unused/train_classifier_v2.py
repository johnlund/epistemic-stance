"""
Epistemic Stance Classifier Training Script v2

Updated based on actual labeling results:
- absolutist: 576 (20.5%)
- evaluativist: 1443 (51.4%)
- multiplist: 111 (4.0%)
- error: 676 (24.1%) - excluded from training

Key changes from v1:
1. Class weighting to handle imbalance (multiplist is 13x rarer than evaluativist)
2. Confidence-based sample weighting (high confidence samples weighted more)
3. Option to use focal loss for hard example mining
4. Stratified sampling to ensure multiplist representation in batches
5. Evaluation metrics focused on per-class performance (especially multiplist recall)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import evaluate
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Model
    model_name = "microsoft/deberta-v3-base"  # Good for nuanced classification
    # Alternative: "roberta-large" for more capacity
    
    # Data
    max_length = 512
    test_size = 0.15
    val_size = 0.15  # From remaining after test split
    min_confidence = "low"  # Include all confidence levels, or "medium"/"high" to filter
    
    # Training
    batch_size = 8  # Small batch for class balance
    gradient_accumulation_steps = 4  # Effective batch = 32
    learning_rate = 2e-5
    num_epochs = 10
    warmup_ratio = 0.1
    weight_decay = 0.01
    
    # Class imbalance handling
    use_class_weights = True
    use_weighted_sampler = True  # Oversample minority classes
    use_focal_loss = False  # Alternative to class weights
    focal_loss_gamma = 2.0
    
    # Sample weighting by confidence
    confidence_weights = {
        "high": 1.0,
        "medium": 0.8,
        "low": 0.5
    }
    
    # Output
    output_dir = "./epistemic_stance_classifier"
    

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data(csv_path: str, config: Config):
    """Load labeled data and prepare for training."""
    
    df = pd.read_csv(csv_path)
    
    # Filter out errors
    df = df[df['epistemic_stance'] != 'error'].copy()
    print(f"Loaded {len(df)} samples (after removing errors)")
    
    # Filter by confidence if specified
    if config.min_confidence == "high":
        df = df[df['stance_confidence'] == 'high']
    elif config.min_confidence == "medium":
        df = df[df['stance_confidence'].isin(['high', 'medium'])]
    
    print(f"After confidence filter: {len(df)} samples")
    print(f"\nStance distribution:")
    print(df['epistemic_stance'].value_counts())
    
    # Create label mapping
    label2id = {'absolutist': 0, 'multiplist': 1, 'evaluativist': 2}
    id2label = {v: k for k, v in label2id.items()}
    
    df['label'] = df['epistemic_stance'].map(label2id)
    
    # Create sample weights based on confidence
    df['sample_weight'] = df['stance_confidence'].map(config.confidence_weights)
    
    return df, label2id, id2label


def create_splits(df: pd.DataFrame, config: Config):
    """Create stratified train/val/test splits."""
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=config.test_size,
        stratify=df['label'],
        random_state=42
    )
    
    # Second split: separate validation from training
    val_ratio = config.val_size / (1 - config.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df['label'],
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    print(f"\nTrain distribution:")
    print(train_df['epistemic_stance'].value_counts())
    
    return train_df, val_df, test_df


# =============================================================================
# CLASS IMBALANCE HANDLING
# =============================================================================

def compute_class_weights_tensor(df: pd.DataFrame, label2id: dict):
    """Compute class weights inversely proportional to frequency."""
    
    labels = df['label'].values
    classes = np.array(list(label2id.values()))
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    print(f"\nComputed class weights:")
    for label_name, label_id in label2id.items():
        print(f"  {label_name}: {weights[label_id]:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(df: pd.DataFrame, label2id: dict):
    """Create a sampler that oversamples minority classes."""
    
    # Count samples per class
    class_counts = df['label'].value_counts().sort_index()
    
    # Weight for each sample = 1 / (count of its class)
    sample_weights = df['label'].apply(lambda x: 1.0 / class_counts[x])
    
    # Optionally multiply by confidence weight
    sample_weights = sample_weights * df['sample_weight']
    
    sampler = WeightedRandomSampler(
        weights=sample_weights.values,
        num_samples=len(df),
        replacement=True
    )
    
    return sampler


# =============================================================================
# FOCAL LOSS (OPTIONAL - ALTERNATIVE TO CLASS WEIGHTS)
# =============================================================================

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples by down-weighting easy ones.
    """
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, 
            weight=self.class_weights,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =============================================================================
# DATASET AND TOKENIZATION
# =============================================================================

class EpistemicStanceDataset(Dataset):
    """PyTorch dataset for epistemic stance classification."""
    
    def __init__(self, df, tokenizer, max_length):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.sample_weights = df['sample_weight'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'sample_weight': torch.tensor(self.sample_weights[idx], dtype=torch.float)
        }


def tokenize_for_hf(examples, tokenizer, max_length):
    """Tokenization function for HuggingFace datasets."""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )


# =============================================================================
# CUSTOM TRAINER WITH CLASS WEIGHTS
# =============================================================================

class WeightedTrainer(Trainer):
    """Trainer that supports class weights in loss computation."""
    
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        if use_focal_loss:
            self.focal_loss = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels)
        elif self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, labels, 
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred):
    """Compute metrics with focus on per-class performance."""
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Overall metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    # Per-class F1 (important for imbalanced data)
    f1_per_class = f1_metric.compute(predictions=predictions, references=labels, average=None)
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1_macro': f1_macro['f1'],
        'f1_weighted': f1_weighted['f1'],
        'f1_absolutist': f1_per_class['f1'][0],
        'f1_multiplist': f1_per_class['f1'][1],  # Key metric!
        'f1_evaluativist': f1_per_class['f1'][2],
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_classifier(csv_path: str, config: Config = None):
    """Main training pipeline."""
    
    if config is None:
        config = Config()
    
    # Load and prepare data
    df, label2id, id2label = load_and_prepare_data(csv_path, config)
    train_df, val_df, test_df = create_splits(df, config)
    
    # Compute class weights
    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights_tensor(train_df, label2id)
    
    # Load tokenizer and model
    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    
    # Create HuggingFace datasets
    train_dataset = HFDataset.from_pandas(train_df[['text', 'label']])
    val_dataset = HFDataset.from_pandas(val_df[['text', 'label']])
    test_dataset = HFDataset.from_pandas(test_df[['text', 'label']])
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_for_hf(x, tokenizer, config.max_length),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_for_hf(x, tokenizer, config.max_length),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_for_hf(x, tokenizer, config.max_length),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_multiplist",  # Optimize for multiplist detection!
        greater_is_better=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        report_to="none",  # Disable wandb etc.
        seed=42,
    )
    
    # Create trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_loss_gamma,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Detailed classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=['absolutist', 'multiplist', 'evaluativist']
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(pd.DataFrame(
        cm,
        index=['true_absolutist', 'true_multiplist', 'true_evaluativist'],
        columns=['pred_absolutist', 'pred_multiplist', 'pred_evaluativist']
    ))
    
    # Save model
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    print(f"\nModel saved to {config.output_dir}/final")
    
    return trainer, test_results


# =============================================================================
# INFERENCE
# =============================================================================

def predict_stance(text: str, model_path: str):
    """Predict epistemic stance for a single text."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
    
    id2label = model.config.id2label
    
    return {
        'predicted_stance': id2label[pred_class],
        'probabilities': {
            id2label[i]: probs[0][i].item() 
            for i in range(len(id2label))
        }
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train epistemic stance classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled CSV")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss instead of class weights")
    parser.add_argument("--min-confidence", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument("--output", type=str, default="./epistemic_stance_classifier")
    
    args = parser.parse_args()
    
    config = Config()
    config.model_name = args.model
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.use_focal_loss = args.focal_loss
    config.use_class_weights = not args.focal_loss  # Use one or the other
    config.min_confidence = args.min_confidence
    config.output_dir = args.output
    
    train_classifier(args.data, config)
