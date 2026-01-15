#!/usr/bin/env python3
"""
Classifier Error Analysis Script
================================

Analyzes classifier predictions to identify:
- Systematic error patterns
- Confusing sample characteristics  
- Per-class performance issues
- Samples to potentially re-label

Author: Claude (Anthropic)
Project: Epistemic Stance Analysis Pipeline
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """Load test predictions from training output."""
    df = pd.read_csv(predictions_path)
    
    # Ensure we have the required columns
    required = ['text', 'epistemic_stance', 'predicted_label']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    # Add error indicator
    df['is_error'] = df['epistemic_stance'] != df['predicted_label']
    df['error_type'] = df.apply(
        lambda row: f"{row['epistemic_stance']}_as_{row['predicted_label']}" 
        if row['is_error'] else 'correct',
        axis=1
    )
    
    return df


def analyze_errors(df: pd.DataFrame) -> Dict:
    """Comprehensive error analysis."""
    
    results = {}
    
    # Overall accuracy
    accuracy = (df['epistemic_stance'] == df['predicted_label']).mean()
    results['overall_accuracy'] = accuracy
    
    # Error counts by type
    error_df = df[df['is_error']]
    error_types = error_df['error_type'].value_counts().to_dict()
    results['error_type_counts'] = error_types
    
    # Per-class accuracy
    per_class = {}
    for label in df['epistemic_stance'].unique():
        subset = df[df['epistemic_stance'] == label]
        acc = (subset['predicted_label'] == label).mean()
        per_class[label] = {
            'accuracy': acc,
            'total': len(subset),
            'correct': (subset['predicted_label'] == label).sum(),
            'errors': (subset['predicted_label'] != label).sum()
        }
    results['per_class_accuracy'] = per_class
    
    # Confidence analysis for errors
    if 'predicted_prob_absolutist' in df.columns:
        error_conf = df[df['is_error']]['silver_confidence' if 'silver_confidence' in df.columns 
                                        else df.columns[df.columns.str.contains('prob')].tolist()[0]]
        correct_conf = df[~df['is_error']]['silver_confidence' if 'silver_confidence' in df.columns
                                           else df.columns[df.columns.str.contains('prob')].tolist()[0]]
        
        # Get max probability for each sample
        prob_cols = [c for c in df.columns if c.startswith('predicted_prob_')]
        if prob_cols:
            df['max_prob'] = df[prob_cols].max(axis=1)
            results['confidence_analysis'] = {
                'mean_confidence_errors': float(df[df['is_error']]['max_prob'].mean()),
                'mean_confidence_correct': float(df[~df['is_error']]['max_prob'].mean()),
                'errors_above_0.8': int((df[df['is_error']]['max_prob'] > 0.8).sum()),
                'errors_above_0.9': int((df[df['is_error']]['max_prob'] > 0.9).sum()),
            }
    
    # Word count analysis
    if 'word_count' in df.columns:
        results['word_count_analysis'] = {
            'mean_wc_errors': float(error_df['word_count'].mean()),
            'mean_wc_correct': float(df[~df['is_error']]['word_count'].mean()),
            'errors_above_400_words': int((error_df['word_count'] > 400).sum()),
        }
    
    # Most common error patterns
    results['most_common_errors'] = list(error_types.items())[:5]
    
    return results


def find_confusing_samples(
    df: pd.DataFrame,
    n_samples: int = 20
) -> Dict[str, pd.DataFrame]:
    """Find samples that are most likely to confuse the classifier."""
    
    confusing = {}
    
    # High-confidence errors (classifier was confident but wrong)
    prob_cols = [c for c in df.columns if c.startswith('predicted_prob_')]
    if prob_cols:
        df['max_prob'] = df[prob_cols].max(axis=1)
        high_conf_errors = df[df['is_error']].nlargest(n_samples, 'max_prob')
        confusing['high_confidence_errors'] = high_conf_errors
    
    # Low-confidence correct (classifier was uncertain but right)
    if prob_cols:
        low_conf_correct = df[~df['is_error']].nsmallest(n_samples, 'max_prob')
        confusing['low_confidence_correct'] = low_conf_correct
    
    # Errors by type
    error_df = df[df['is_error']]
    for error_type in error_df['error_type'].unique():
        subset = error_df[error_df['error_type'] == error_type].head(n_samples)
        confusing[f'examples_{error_type}'] = subset
    
    return confusing


def extract_linguistic_patterns(df: pd.DataFrame) -> Dict:
    """Extract common linguistic patterns in errors vs correct predictions."""
    
    import re
    
    patterns = {
        'hedging': r'\b(maybe|perhaps|might|could be|possibly|I think|seems|appears)\b',
        'certainty': r'\b(definitely|certainly|absolutely|always|never|must|clearly)\b',
        'relativism': r'\b(depends on|up to you|everyone|personal|subjective|opinion)\b',
        'evidence': r'\b(research|studies|evidence|data|shows|proves|according to)\b',
        'counterargument': r'\b(however|although|but|on the other hand|while|despite)\b',
    }
    
    results = {'errors': {}, 'correct': {}}
    
    error_texts = df[df['is_error']]['text'].tolist()
    correct_texts = df[~df['is_error']]['text'].tolist()
    
    for pattern_name, pattern in patterns.items():
        # Count in errors
        error_count = sum(1 for t in error_texts if re.search(pattern, t, re.IGNORECASE))
        error_rate = error_count / len(error_texts) if error_texts else 0
        
        # Count in correct
        correct_count = sum(1 for t in correct_texts if re.search(pattern, t, re.IGNORECASE))
        correct_rate = correct_count / len(correct_texts) if correct_texts else 0
        
        results['errors'][pattern_name] = {'count': error_count, 'rate': error_rate}
        results['correct'][pattern_name] = {'count': correct_count, 'rate': correct_rate}
        
        # Flag patterns that differ significantly
        if abs(error_rate - correct_rate) > 0.1:
            results[f'{pattern_name}_significant'] = True
    
    return results


def generate_report(
    df: pd.DataFrame,
    output_dir: str
) -> str:
    """Generate a comprehensive error analysis report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analyses
    error_analysis = analyze_errors(df)
    confusing_samples = find_confusing_samples(df)
    linguistic_patterns = extract_linguistic_patterns(df)
    
    # Build report
    report_lines = [
        "# Classifier Error Analysis Report",
        "=" * 50,
        "",
        "## Overall Performance",
        f"- Accuracy: {error_analysis['overall_accuracy']:.2%}",
        f"- Total samples: {len(df)}",
        f"- Total errors: {df['is_error'].sum()}",
        "",
        "## Per-Class Performance",
    ]
    
    for label, stats in error_analysis['per_class_accuracy'].items():
        report_lines.append(f"### {label.capitalize()}")
        report_lines.append(f"- Accuracy: {stats['accuracy']:.2%}")
        report_lines.append(f"- Correct: {stats['correct']}/{stats['total']}")
        report_lines.append(f"- Errors: {stats['errors']}")
        report_lines.append("")
    
    report_lines.extend([
        "## Error Type Breakdown",
    ])
    
    for error_type, count in error_analysis['error_type_counts'].items():
        report_lines.append(f"- {error_type}: {count}")
    
    if 'confidence_analysis' in error_analysis:
        report_lines.extend([
            "",
            "## Confidence Analysis",
            f"- Mean confidence on errors: {error_analysis['confidence_analysis']['mean_confidence_errors']:.3f}",
            f"- Mean confidence on correct: {error_analysis['confidence_analysis']['mean_confidence_correct']:.3f}",
            f"- High-confidence errors (>0.8): {error_analysis['confidence_analysis']['errors_above_0.8']}",
            f"- Very high-confidence errors (>0.9): {error_analysis['confidence_analysis']['errors_above_0.9']}",
        ])
    
    report_lines.extend([
        "",
        "## Linguistic Pattern Analysis",
        "",
        "Patterns more common in errors vs correct predictions:",
    ])
    
    for pattern in ['hedging', 'certainty', 'relativism', 'evidence', 'counterargument']:
        error_rate = linguistic_patterns['errors'].get(pattern, {}).get('rate', 0)
        correct_rate = linguistic_patterns['correct'].get(pattern, {}).get('rate', 0)
        diff = error_rate - correct_rate
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
        report_lines.append(f"- {pattern}: errors={error_rate:.1%}, correct={correct_rate:.1%} ({direction})")
    
    report_lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    
    # Generate recommendations based on analysis
    if error_analysis['per_class_accuracy'].get('multiplist', {}).get('accuracy', 1) < 0.5:
        report_lines.append("⚠️ **Multiplist accuracy is low.** Consider:")
        report_lines.append("  - Reviewing multiplist training samples for consistency")
        report_lines.append("  - Increasing focal loss gamma")
        report_lines.append("  - Data augmentation for multiplist class")
        report_lines.append("")
    
    if 'confidence_analysis' in error_analysis:
        if error_analysis['confidence_analysis']['errors_above_0.8'] > 10:
            report_lines.append("⚠️ **High-confidence errors detected.** Consider:")
            report_lines.append("  - Reviewing these samples manually")
            report_lines.append("  - Checking for labeling inconsistencies")
            report_lines.append("  - Using higher confidence threshold for silver labels")
            report_lines.append("")
    
    # Common error patterns
    if 'evaluativist_as_multiplist' in error_analysis['error_type_counts']:
        if error_analysis['error_type_counts']['evaluativist_as_multiplist'] > 20:
            report_lines.append("⚠️ **Many evaluativist→multiplist errors.** The model may be:")
            report_lines.append("  - Confusing epistemic humility with relativism")
            report_lines.append("  - Missing evidence-weighing signals")
            report_lines.append("")
    
    report = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(output_dir, 'error_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save detailed data
    json_path = os.path.join(output_dir, 'error_analysis.json')
    with open(json_path, 'w') as f:
        json.dump({
            'error_analysis': error_analysis,
            'linguistic_patterns': linguistic_patterns,
        }, f, indent=2, default=str)
    
    # Save confusing samples for manual review
    for name, samples_df in confusing_samples.items():
        if len(samples_df) > 0:
            samples_path = os.path.join(output_dir, f'{name}.csv')
            samples_df.to_csv(samples_path, index=False)
    
    print(report)
    print(f"\nFull report saved to: {report_path}")
    print(f"Detailed data saved to: {json_path}")
    print(f"Confusing samples saved to: {output_dir}/")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze classifier errors")
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to test_predictions.csv from training')
    parser.add_argument('--output-dir', type=str, default='./error_analysis',
                        help='Output directory for analysis')
    
    args = parser.parse_args()
    
    df = load_predictions(args.predictions)
    generate_report(df, args.output_dir)


if __name__ == '__main__':
    main()
