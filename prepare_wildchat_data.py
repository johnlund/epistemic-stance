#!/usr/bin/env python3
"""
WildChat LLM Data Preparation for Epistemic Stance Training
============================================================

Downloads WildChat dataset, filters for substantive assistant responses,
and prepares for silver labeling with the trained classifier.

Usage:
    # Step 1: Run locally to download and filter WildChat
    python prepare_wildchat_data.py filter --output wildchat_filtered.csv --samples 20000
    
    # Step 2: Upload to Lambda Cloud and run classifier
    python inference_silver_labels.py predict \
        --model ./classifier_output_final/best_model \
        --data wildchat_filtered.csv \
        --output wildchat_silver_labels.csv \
        --threshold 0.85

Author: Claude (Anthropic)
Project: Epistemic Stance Analysis Pipeline
"""

import argparse
import re
import logging
from typing import Optional
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LINGUISTIC PATTERNS (from your existing filters)
# ============================================================================

GENERAL_REASONING_PATTERNS = [
    r'\bbecause\b',
    r'\btherefore\b',
    r'\bhowever\b',
    r'\balthough\b',
    r'\bevidence\b',
    r'\breason\b',
    r'\bargument\b',
    r'\bbelieve\b',
    r'\bthink\b',
    r'\bprove\b',
    r'\bshow[s]?\b',
    r'\bdemonstrate\b',
    r'\bin my (view|opinion)\b',
    r'\bon the other hand\b',
    r'\bfor example\b',
    r'\bfor instance\b',
    r'\baccording to\b',
    r'\bresearch\b',
    r'\bstudy\b',
    r'\bdata\b',
]

ABSOLUTIST_PATTERNS = [
    r'\bobviously\b',
    r'\bclearly\b',
    r'\bclearly (you should|the answer)\b',
    r'\bundeniably\b',
    r'\bthe fact is\b',
    r'\bthe truth is\b',
    r'\bno doubt\b',
    r'\bwithout question\b',
    r'\babsolutely\b',
    r'\bdefinitely\b',
    r'\beveryone knows\b',
    r'\bit\'s clear that\b',
    r'\byou (should|must|need to) (definitely|absolutely)\b',
    r'\bthere\'s no (question|doubt)\b',
]

EVALUATIVIST_PATTERNS = [
    r'\bthe evidence (suggests|shows)\b',
    r'\bon balance\b',
    r'\bwhile.+however\b',
    r'\balthough.+still\b',
    r'\bi could be wrong\b',
    r'\bmore (likely|reasonable|justified)\b',
    r'\bthe (better|best|stronger) (option|choice|argument)\b',
    r'\bstronger argument\b',
    r'\bbetter supported\b',
    r'\bhaving considered\b',
    r'\bweighing\b',
]

MULTIPLIST_STRONG_PATTERNS = [
    r'\bonly you can (decide|know|answer|figure)',
    r'\bthat\'s (really )?for you to decide\b',
    r'\bonly you know\b',
    r'\byou\'re the only one who\b',
    r'\bno one (else )?can (tell|decide|know)\b',
    r'\bit\'s (really )?(all )?subjective\b',
    r'\bit (really )?depends on (the person|you|your|how you)\b',
    r'\beveryone\'s (situation|relationship|circumstances) is different\b',
    r'\bthere\'s no (right|wrong|one|correct) answer\b',
    r'\bno right or wrong (here|answer)\b',
    r'\bboth (are|views?|perspectives?|sides?) (are )?(valid|legitimate|understandable)\b',
    r'\bneither (is|are) (right|wrong)\b',
    r'\beveryone\'s entitled to\b',
    r'\bwho\'s to say\b',
    r'\bwho am i to (judge|say)\b',
    r'\bi (can\'t|cannot|won\'t) (tell you what|say what|judge)\b',
    r'\bnot (for me|my place) to (say|judge|decide)\b',
    r'\bi\'m not (going to|gonna) (tell you|judge|say)\b',
    r'\bboth (could be|are) (true|right|correct)\b',
]

MULTIPLIST_MODERATE_PATTERNS = [
    r'\bit (really )?depends on (the person|who you are|your perspective)\b',
    r'\bthat\'s (just )?your (call|decision|choice)\b',
    r'\byou (have to|need to|gotta) decide\b',
    r'\bup to you\b',
    r'\byour (call|choice|decision)\b',
    r'\bpersonal (choice|preference|decision)\b',
    r'\bdo what (feels|seems) right (to|for) you\b',
    r'\bwhatever (you think|works for you|feels right)\b',
    r'\bjust my (opinion|two cents|perspective)\b',
    r'\beveryone is different\b',
    r'\bdifferent things work for different\b',
    r'\beveryone.+entitled\b',
    r'\bwho\'s to say\b',
    r'\bit\'s subjective\b',
    r'\bboth.+valid\b',
    r'\bneither.+wrong\b',
    r'\bnot for me to judge\b',
]


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def has_reasoning_indicators(text: str) -> bool:
    """
    Check if text contains indicators of epistemic reasoning.
    Requires at least 2 pattern matches.
    """
    text_lower = text.lower()
    
    all_patterns = (
        GENERAL_REASONING_PATTERNS +
        ABSOLUTIST_PATTERNS +
        EVALUATIVIST_PATTERNS +
        MULTIPLIST_STRONG_PATTERNS +
        MULTIPLIST_MODERATE_PATTERNS
    )
    
    matches = sum(1 for pattern in all_patterns if re.search(pattern, text_lower))
    return matches >= 2


def has_multiplist_indicators(text: str) -> bool:
    """Check if text contains multiplist indicators."""
    text_lower = text.lower()
    all_patterns = MULTIPLIST_STRONG_PATTERNS + MULTIPLIST_MODERATE_PATTERNS
    
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def is_suitable_for_labeling(
    text: str,
    min_words: int = 100,
    max_words: int = 2000,
    require_multiplist_patterns: bool = False
) -> bool:
    """
    Filter text for suitability for epistemic stance labeling.
    
    We want responses that:
    - Are substantive (not too short/long)
    - Contain reasoning indicators
    - Are not primarily code, lists, or structured data
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    word_count = len(text.split())
    
    # Length filters
    if word_count < min_words or word_count > max_words:
        return False
    
    # Skip responses that are mostly code
    code_indicators = text.count('```') + text.count('def ') + text.count('function ')
    if code_indicators > 2:
        return False
    
    # Skip responses that are mostly lists/bullet points
    bullet_count = text.count('\n- ') + text.count('\n* ') + text.count('\n1.')
    if bullet_count > 10:
        return False
    
    # Check for reasoning indicators
    if not has_reasoning_indicators(text):
        return False
    
    # Filter out pure Q&A or very question-heavy responses
    sentences = text.split('.')
    question_ratio = sum(1 for s in sentences if '?' in s) / max(len(sentences), 1)
    if question_ratio > 0.7:
        return False
    
    # Optional multiplist filter for oversampling
    if require_multiplist_patterns:
        if not has_multiplist_indicators(text):
            return False
    
    return True


# ============================================================================
# WILDCHAT PROCESSING
# ============================================================================

def load_wildchat(max_conversations: Optional[int] = None):
    """
    Load WildChat dataset from Hugging Face.
    
    Returns iterator over conversations.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    logger.info("Loading WildChat dataset from Hugging Face...")
    logger.info("(This may take a while on first download)")
    
    # Load dataset - WildChat is large, so we stream it
    dataset = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=True
    )
    
    if max_conversations:
        dataset = dataset.take(max_conversations)
    
    return dataset


def extract_assistant_responses(conversation: dict) -> list:
    """
    Extract assistant responses from a WildChat conversation.
    
    Returns list of dicts with text and metadata.
    """
    responses = []
    
    # WildChat format: conversation is a list of turns
    turns = conversation.get('conversation', [])
    
    for i, turn in enumerate(turns):
        if turn.get('role') == 'assistant':
            content = turn.get('content', '')
            
            if content:
                responses.append({
                    'text': content,
                    'conversation_id': conversation.get('conversation_hash', ''),
                    'turn_index': i,
                    'model': conversation.get('model', 'unknown'),
                    'language': conversation.get('language', 'unknown'),
                })
    
    return responses


def filter_wildchat(
    output_path: str,
    target_samples: int = 20000,
    multiplist_oversample: int = 2000,
    max_conversations: int = 200000,
    min_words: int = 100,
    max_words: int = 1500,
):
    """
    Filter WildChat for suitable epistemic stance labeling samples.
    
    Strategy:
    1. General filter: Get diverse samples with reasoning indicators
    2. Multiplist oversample: Extra pass to find multiplist-like responses
    """
    
    logger.info(f"Target: {target_samples} general + {multiplist_oversample} multiplist-enriched")
    
    dataset = load_wildchat(max_conversations=max_conversations)
    
    general_samples = []
    multiplist_samples = []
    
    conversations_processed = 0
    
    logger.info("Processing conversations...")
    
    for conversation in dataset:
        conversations_processed += 1
        
        if conversations_processed % 10000 == 0:
            logger.info(
                f"Processed {conversations_processed} conversations, "
                f"found {len(general_samples)} general + {len(multiplist_samples)} multiplist"
            )
        
        # Skip non-English for consistency
        if conversation.get('language', 'unknown') != 'English':
            continue
        
        # Extract assistant responses
        responses = extract_assistant_responses(conversation)
        
        for response in responses:
            text = response['text']
            
            # Check for multiplist indicators first (for oversampling)
            if len(multiplist_samples) < multiplist_oversample:
                if is_suitable_for_labeling(text, min_words, max_words, require_multiplist_patterns=True):
                    response['sample_id'] = f"wc_m_{len(multiplist_samples)}"
                    response['filter_type'] = 'multiplist_enriched'
                    multiplist_samples.append(response)
                    continue
            
            # General filter
            if len(general_samples) < target_samples:
                if is_suitable_for_labeling(text, min_words, max_words, require_multiplist_patterns=False):
                    response['sample_id'] = f"wc_g_{len(general_samples)}"
                    response['filter_type'] = 'general'
                    general_samples.append(response)
        
        # Check if we have enough
        if len(general_samples) >= target_samples and len(multiplist_samples) >= multiplist_oversample:
            break
    
    logger.info(f"Final counts: {len(general_samples)} general + {len(multiplist_samples)} multiplist")
    
    # Combine and deduplicate
    all_samples = general_samples + multiplist_samples
    
    # Convert to DataFrame
    df = pd.DataFrame(all_samples)
    
    # Remove duplicates (same response appearing in both lists)
    df = df.drop_duplicates(subset=['text'])
    
    logger.info(f"After deduplication: {len(df)} samples")
    
    # Show distribution by filter type
    logger.info(f"Filter type distribution:\n{df['filter_type'].value_counts()}")
    
    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Print sample statistics
    logger.info("\nSample statistics:")
    logger.info(f"  Mean word count: {df['text'].str.split().str.len().mean():.1f}")
    logger.info(f"  Median word count: {df['text'].str.split().str.len().median():.1f}")
    logger.info(f"  Models represented: {df['model'].nunique()}")
    
    return df


# ============================================================================
# POST-LABELING: COMBINE WITH REDDIT DATA
# ============================================================================

def combine_with_reddit_silver(
    reddit_silver_path: str,
    wildchat_silver_path: str,
    output_path: str,
    wildchat_threshold: float = 0.85,
    wildchat_multiplist_threshold: float = 0.70,
):
    """
    Combine Reddit and WildChat silver labels into unified training set.
    """
    
    logger.info("Loading Reddit silver labels...")
    reddit = pd.read_csv(reddit_silver_path)
    
    logger.info("Loading WildChat silver labels...")
    wildchat = pd.read_csv(wildchat_silver_path)
    
    # Filter WildChat by confidence (with lower threshold for multiplist)
    wc_high_conf = wildchat[
        (wildchat['silver_label'] != 'multiplist') & 
        (wildchat['silver_confidence'] >= wildchat_threshold)
    ]
    wc_multiplist = wildchat[
        (wildchat['silver_label'] == 'multiplist') & 
        (wildchat['silver_confidence'] >= wildchat_multiplist_threshold)
    ]
    wildchat_filtered = pd.concat([wc_high_conf, wc_multiplist])
    
    logger.info(f"WildChat after filtering: {len(wildchat_filtered)}")
    logger.info(f"WildChat label distribution:\n{wildchat_filtered['silver_label'].value_counts()}")
    
    # Standardize columns
    reddit['data_source'] = 'reddit_silver'
    wildchat_filtered['data_source'] = 'llm_silver'
    
    # Ensure both have required columns
    reddit['label'] = reddit['silver_label'] if 'silver_label' in reddit.columns else reddit['label']
    wildchat_filtered['label'] = wildchat_filtered['silver_label']
    
    reddit['sample_weight'] = reddit['silver_confidence'] if 'silver_confidence' in reddit.columns else reddit['sample_weight']
    wildchat_filtered['sample_weight'] = wildchat_filtered['silver_confidence']
    
    # Select common columns
    cols = ['sample_id', 'text', 'label', 'data_source', 'sample_weight']
    
    reddit_subset = reddit[cols]
    wildchat_subset = wildchat_filtered[cols]
    
    # Combine
    combined = pd.concat([reddit_subset, wildchat_subset], ignore_index=True)
    
    logger.info(f"\nCombined dataset: {len(combined)} samples")
    logger.info(f"By source:\n{combined['data_source'].value_counts()}")
    logger.info(f"By label:\n{combined['label'].value_counts()}")
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    return combined


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare WildChat data for epistemic stance labeling")
    subparsers = parser.add_subparsers(dest='command')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter WildChat for labeling')
    filter_parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    filter_parser.add_argument('--samples', type=int, default=20000, help='Target general samples')
    filter_parser.add_argument('--multiplist-samples', type=int, default=2000, 
                               help='Target multiplist-enriched samples')
    filter_parser.add_argument('--max-conversations', type=int, default=200000,
                               help='Max conversations to process')
    filter_parser.add_argument('--min-words', type=int, default=100, help='Min words per response')
    filter_parser.add_argument('--max-words', type=int, default=1500, help='Max words per response')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine Reddit and WildChat silver labels')
    combine_parser.add_argument('--reddit', type=str, required=True, help='Reddit silver labels CSV')
    combine_parser.add_argument('--wildchat', type=str, required=True, help='WildChat silver labels CSV')
    combine_parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    combine_parser.add_argument('--threshold', type=float, default=0.85, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.command == 'filter':
        filter_wildchat(
            output_path=args.output,
            target_samples=args.samples,
            multiplist_oversample=args.multiplist_samples,
            max_conversations=args.max_conversations,
            min_words=args.min_words,
            max_words=args.max_words,
        )
    
    elif args.command == 'combine':
        combine_with_reddit_silver(
            reddit_silver_path=args.reddit,
            wildchat_silver_path=args.wildchat,
            output_path=args.output,
            wildchat_threshold=args.threshold,
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
