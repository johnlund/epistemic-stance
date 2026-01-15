#!/usr/bin/env python3
"""
Rebalance training data to preserve multiplist representation
while maintaining good coverage of evaluativist/absolutist distinction.

Strategy:
1. Keep ALL multiplist samples (they're precious)
2. Apply sample weights to multiplist (5-10x)
3. Downsample evaluativist to reduce dominance
4. Maintain gold data priority with higher weights
"""

import pandas as pd
import argparse
from pathlib import Path


def load_and_analyze(input_path: str) -> pd.DataFrame:
    """Load dataset and print current distribution."""
    df = pd.read_csv(input_path)
    
    print("=" * 60)
    print("CURRENT DISTRIBUTION")
    print("=" * 60)
    
    print("\nBy data source:")
    print(df['data_source'].value_counts())
    
    print("\nBy label:")
    print(df['label'].value_counts())
    
    print("\nBy source and label:")
    print(pd.crosstab(df['data_source'], df['label']))
    
    return df


def rebalance_dataset(
    df: pd.DataFrame,
    target_multiplist_pct: float = 0.05,
    max_evaluativist_ratio: float = 3.0,
    multiplist_weight: float = 5.0,
    gold_weight: float = 2.0,
    silver_weight: float = 1.0
) -> pd.DataFrame:
    """
    Rebalance dataset to improve multiplist representation.
    
    Args:
        df: Input dataframe with 'label', 'data_source', 'sample_weight' columns
        target_multiplist_pct: Target percentage for multiplist (default 5%)
        max_evaluativist_ratio: Max ratio of evaluativist to absolutist (default 3:1)
        multiplist_weight: Sample weight multiplier for multiplist samples
        gold_weight: Sample weight for gold-labeled data
        silver_weight: Sample weight for silver-labeled data
    
    Returns:
        Rebalanced dataframe
    """
    
    # Separate by label
    multiplist = df[df['label'] == 'multiplist'].copy()
    absolutist = df[df['label'] == 'absolutist'].copy()
    evaluativist = df[df['label'] == 'evaluativist'].copy()
    
    print(f"\nStarting counts:")
    print(f"  Multiplist: {len(multiplist)}")
    print(f"  Absolutist: {len(absolutist)}")
    print(f"  Evaluativist: {len(evaluativist)}")
    
    # Keep ALL multiplist - they're too rare to throw away
    n_multiplist = len(multiplist)
    
    # Calculate target sizes based on multiplist floor
    # If we want multiplist to be ~5%, and we have 323 multiplist samples:
    # Total should be ~6,460 to hit 5%
    # But that's too aggressive a reduction. Instead, let's:
    # 1. Keep all multiplist
    # 2. Keep all absolutist (they're already underrepresented)
    # 3. Downsample evaluativist to max_evaluativist_ratio * absolutist
    
    n_absolutist = len(absolutist)
    max_evaluativist = int(n_absolutist * max_evaluativist_ratio)
    
    print(f"\nRebalancing strategy:")
    print(f"  Keep all {n_multiplist} multiplist samples")
    print(f"  Keep all {n_absolutist} absolutist samples")
    print(f"  Cap evaluativist at {max_evaluativist_ratio}x absolutist = {max_evaluativist}")
    
    # Downsample evaluativist, prioritizing gold data
    if len(evaluativist) > max_evaluativist:
        # Separate gold and silver evaluativist
        eval_gold = evaluativist[evaluativist['data_source'] == 'gold']
        eval_silver = evaluativist[evaluativist['data_source'] != 'gold']
        
        # Keep all gold, sample from silver
        n_silver_needed = max_evaluativist - len(eval_gold)
        
        if n_silver_needed > 0 and len(eval_silver) > n_silver_needed:
            eval_silver_sampled = eval_silver.sample(n=n_silver_needed, random_state=42)
            evaluativist_final = pd.concat([eval_gold, eval_silver_sampled])
        else:
            evaluativist_final = evaluativist.head(max_evaluativist)
        
        print(f"  Downsampled evaluativist: {len(evaluativist)} -> {len(evaluativist_final)}")
    else:
        evaluativist_final = evaluativist
    
    # Combine
    rebalanced = pd.concat([multiplist, absolutist, evaluativist_final], ignore_index=True)
    
    # Apply sample weights
    def compute_weight(row):
        base_weight = gold_weight if row['data_source'] == 'gold' else silver_weight
        label_multiplier = multiplist_weight if row['label'] == 'multiplist' else 1.0
        return base_weight * label_multiplier
    
    rebalanced['sample_weight'] = rebalanced.apply(compute_weight, axis=1)
    
    # Shuffle
    rebalanced = rebalanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return rebalanced


def print_final_stats(df: pd.DataFrame):
    """Print final distribution and effective weights."""
    print("\n" + "=" * 60)
    print("REBALANCED DISTRIBUTION")
    print("=" * 60)
    
    print("\nBy label:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print(f"\nPercentages:")
    print((label_counts / len(df) * 100).round(1))
    
    print("\nBy data source:")
    print(df['data_source'].value_counts())
    
    print("\nBy source and label:")
    print(pd.crosstab(df['data_source'], df['label']))
    
    print("\nSample weights by label:")
    print(df.groupby('label')['sample_weight'].agg(['mean', 'sum']))
    
    # Effective representation (accounting for weights)
    print("\nEffective representation (weight-adjusted):")
    weight_by_label = df.groupby('label')['sample_weight'].sum()
    total_weight = weight_by_label.sum()
    print((weight_by_label / total_weight * 100).round(1))


def main():
    parser = argparse.ArgumentParser(description='Rebalance training data for epistemic stance classifier')
    parser.add_argument('--input', '-i', required=True, help='Input CSV path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--target-multiplist', type=float, default=0.05,
                        help='Target multiplist percentage (default: 0.05)')
    parser.add_argument('--max-eval-ratio', type=float, default=3.0,
                        help='Max evaluativist:absolutist ratio (default: 3.0)')
    parser.add_argument('--multiplist-weight', type=float, default=5.0,
                        help='Sample weight multiplier for multiplist (default: 5.0)')
    parser.add_argument('--gold-weight', type=float, default=2.0,
                        help='Base sample weight for gold data (default: 2.0)')
    
    args = parser.parse_args()
    
    # Load and analyze
    df = load_and_analyze(args.input)
    
    # Rebalance
    rebalanced = rebalance_dataset(
        df,
        target_multiplist_pct=args.target_multiplist,
        max_evaluativist_ratio=args.max_eval_ratio,
        multiplist_weight=args.multiplist_weight,
        gold_weight=args.gold_weight
    )
    
    # Print final stats
    print_final_stats(rebalanced)
    
    # Save
    rebalanced.to_csv(args.output, index=False)
    print(f"\nSaved rebalanced dataset to: {args.output}")
    print(f"Total samples: {len(rebalanced)}")


if __name__ == '__main__':
    main()
