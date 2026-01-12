"""
Reddit Subreddit Data Exploration and Preparation
==================================================

This script downloads and filters subreddit data from HuggingFaceGECLM/REDDIT_comments
to find multiplist reasoning examples. Can work with any subreddit split in the dataset.

Multiplist reasoning characteristics:
- Treats knowledge as subjective/personal
- Avoids evaluating which perspective is better
- Deflects judgment: "only you can decide", "it depends on the person"
- Treats all opinions as equally valid

This contrasts with:
- Absolutist: "You should definitely do X, there's no question"
- Evaluativist: "While I see both sides, X seems better because..."

Requirements:
    pip install datasets pandas tqdm

Usage:
    python explore_reddit_ds.py [split_name]
    
    Examples:
    python explore_reddit_ds.py relationship_advice
    python explore_reddit_ds.py changemyview
    python explore_reddit_ds.py AskHistorians
"""

import pandas as pd
from collections import Counter
import json
import random
from tqdm import tqdm
import re

# ============================================================================
# STEP 1: Load a Reddit subreddit split
# ============================================================================

def load_reddit_split(split_name='relationship_advice'):
    """
    Load a specific subreddit split from HuggingFaceGECLM/REDDIT_comments.
    
    This dataset contains Reddit comments from 50 high-quality subreddits,
    each available as a separate split.
    
    Args:
        split_name: Name of the subreddit split to load (e.g., 'relationship_advice', 
                   'changemyview', 'AskHistorians', etc.)
    
    Returns:
        Dataset split for the specified subreddit
    """
    from datasets import load_dataset
    
    print(f"Loading {split_name} split from HuggingFaceGECLM/REDDIT_comments...")
    print("(This may take a few minutes on first download)")
    
    # Load the default config - the dataset has splits for each subreddit
    ds = load_dataset(
        "HuggingFaceGECLM/REDDIT_comments",
        "default"
    )
    
    print(f"\nLoaded dataset with splits: {list(ds.keys())}")
    
    # Check if the requested split exists
    if split_name not in ds:
        available = list(ds.keys())
        print(f"\n❌ Error: Split '{split_name}' not found.")
        print(f"Available splits: {available}")
        raise ValueError(f"Split '{split_name}' not found. Available: {available}")
    
    # Get the requested split
    split_ds = ds[split_name]
    print(f"\nUsing split: {split_name}")
    print(f"Number of comments: {len(split_ds)}")
    print(f"Columns: {split_ds.column_names}")
    
    return split_ds

def explore_structure(ds, n_samples=5):
    """Examine the structure of the dataset."""
    
    print("\n" + "="*60)
    print("DATASET STRUCTURE")
    print("="*60)
    
    print(f"\nColumns: {ds.column_names}")
    
    print("\n--- Sample entries ---")
    for i in range(min(n_samples, len(ds))):
        entry = ds[i]
        print(f"\n[Sample {i+1}]")
        for key, value in entry.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")


# ============================================================================
# STEP 2: Define multiplist linguistic patterns
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
    r'\bobviously\b', r'\bclearly\b', r'\bundeniably\b',
    r'\bthe fact is\b', r'\bthe truth is\b', r'\bno doubt\b',
    r'\bwithout question\b', r'\babsolutely\b', r'\bdefinitely\b',
    r'\beveryone knows\b', r'\bit\'s clear that\b',

    # Absolutist markers (certainty)
    r'\byou (should|must|need to) (definitely|absolutely)\b',
    r'\bthere\'s no (question|doubt)\b',
    r'\bobviously\b',
    r'\bclearly (you should|the answer)\b',
    r'\b(dump|leave|divorce) (him|her|them)\b',  # Strong directive advice
    r'\bred flag\b',  # Definitive judgment
    r'\bdeal ?breaker\b',
]

EVALUATIVIST_PATTERNS = [
    r'\bthe evidence suggests\b', r'\bon balance\b',
    r'\bwhile.+however\b', r'\balthough.+still\b',
    r'\bi could be wrong\b', r'\bmore likely\b',
    r'\bstronger argument\b', r'\bbetter supported\b',
    r'\bhaving considered\b', r'\bweighing\b',
    r'\bI\'ve changed my mind\b', r'\byou\'ve convinced me\b',

    # Evaluativist markers (weighing evidence)
    r'\bthe (better|best|stronger) (option|choice|argument)\b',
    r'\bon balance\b',
    r'\bweighing\b',
    r'\bmore (likely|reasonable|justified)\b',
    r'\bthe evidence (suggests|shows)\b',
]

# Strong multiplist indicators - phrases that suggest relativistic thinking
MULTIPLIST_STRONG_PATTERNS = [
    # Deflecting judgment
    r'\bonly you can (decide|know|answer|figure)',
    r'\bthat\'s (really )?for you to decide\b',
    r'\bonly you know\b',
    r'\byou\'re the only one who\b',
    r'\bno one (else )?can (tell|decide|know)\b',
    
    # Subjectivity framing
    r'\bit\'s (really )?(all )?subjective\b',
    r'\bit (really )?depends on (the person|you|your|how you)\b',
    r'\beveryone\'s (situation|relationship|circumstances) is different\b',
    r'\bthere\'s no (right|wrong|one|correct) answer\b',
    r'\bno right or wrong (here|answer)\b',
    
    # Equal validity
    r'\bboth (are|views?|perspectives?|sides?) (are )?(valid|legitimate|understandable)\b',
    r'\bneither (is|are) (right|wrong)\b',
    r'\beveryone\'s entitled to\b',
    r'\bwho\'s to say\b',
    r'\bwho am i to (judge|say)\b',
    
    # Refusing to evaluate
    r'\bi (can\'t|cannot|won\'t) (tell you what|say what|judge)\b',
    r'\bnot (for me|my place) to (say|judge|decide)\b',
    r'\bi\'m not (going to|gonna) (tell you|judge|say)\b',
]

# Moderate multiplist indicators - softer versions
MULTIPLIST_MODERATE_PATTERNS = [
    r'\bit (really )?depends\b',
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
    r'\bjust my opinion\b',
    r'\beveryone.+entitled\b',
    r'\bwho\'s to say\b',
    r'\bit\'s subjective\b',
    r'\bdepends on the person\b',
    r'\bboth.+valid\b',
    r'\bneither.+wrong\b',
    r'\bnot for me to judge\b',

    r'\bjust my opinion\b', r'\beveryone.+entitled\b', 
    r'\bwho\'s to say\b', r'\bit\'s subjective\b',
    r'\bdepends on the person\b', r'\bboth.+valid\b',
    r'\bneither.+wrong\b', r'\bnot for me to judge\b',
]

def has_multiplist_indicators(text):
    """
    Check if text contains any multiplist indicators.
    
    Returns True if any pattern from MULTIPLIST_STRONG_PATTERNS,
    MULTIPLIST_MODERATE_PATTERNS, or MULTIPLIST_INDICATORS matches.
    """
    text_lower = text.lower()
    
    # Combine all patterns
    all_patterns = MULTIPLIST_STRONG_PATTERNS + MULTIPLIST_MODERATE_PATTERNS
    
    # Check if any pattern matches
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def has_reasoning_indicators(text):
    """
    Check if text contains indicators of epistemic reasoning.
    
    We want posts where people are actually reasoning about claims,
    not just stating preferences or making jokes.
    """
    
    text_lower = text.lower()
    matches = sum(1 for pattern in GENERAL_REASONING_PATTERNS if re.search(pattern, text_lower))
    
    return matches >= 2  # At least 2 reasoning indicators


# def filter_utterance_for_epistemic_labeling(text, min_words=50, max_words=None):
#     """
#     Filter a single utterance for suitability for epistemic stance labeling.
    
#     Args:
#         text: The utterance text
#         min_words: Minimum word count (need enough content to assess stance)
#         max_words: Maximum word count (for practical labeling)
    
#     Returns: (is_suitable, reason)
#     """
#     word_count = len(text.split())
    
#     # Length filters
#     if word_count < min_words:
#         return False, f"too_short ({word_count} words)"
    
#     # No max limit for labeling stage
#     # Long posts often contain the richest epistemic reasoning
#     # if word_count > max_words:
#     #     return False, f"too_long ({word_count} words)"
    
#     # Check for deleted/removed content
#     if '[deleted]' in text or '[removed]' in text:
#         return False, "deleted_content"
    
#     # Check for reasoning indicators
#     if not has_reasoning_indicators(text):
#         return False, "no_reasoning_indicators"
    
#     # Filter out pure questions without argumentation
#     sentences = text.split('.')
#     question_ratio = sum(1 for s in sentences if '?' in s) / max(len(sentences), 1)
#     if question_ratio > 0.7:
#         return False, "mostly_questions"
    
#     # Filter out very short sentences (likely not substantive)
#     avg_sentence_length = word_count / max(len(sentences), 1)
#     if avg_sentence_length < 8:
#         return False, "choppy_writing"
    
#     return True, "suitable"

# ============================================================================
# STEP 3: Filter for suitable samples
# ============================================================================

def get_text_field(entry):
    """Extract the text content from an entry, handling different possible field names."""
    # Try common field names
    for field in ['body']:
        if field in entry and entry[field]:
            return entry[field]
    return None


def filter_for_labeling(entry, min_words=100, max_words=2000, require_multiplist_patterns=False):
    """
    Filter a single entry for suitability for labeling.
    
    We want comments that:
    - Are substantive (not too short)
    - Contain reasoning (not just "lol" or "this")
    - Optionally: Show multiplist indicators (if require_multiplist_patterns=True)
    - Don't have strong absolutist/evaluativist markers
    
    Args:
        entry: Dataset entry
        min_words: Minimum word count
        max_words: Maximum word count
        require_multiplist_patterns: If True, only include entries with multiplist indicators
    
    Returns: bool - True if suitable, False otherwise
    """
    text = get_text_field(entry)
    
    if not text:
        return False
    
    # Clean up text
    text = text.strip()
    
    # Basic filters
    word_count = len(text.split())
    
    if word_count < min_words:
        return False
    
    if word_count > max_words:
        return False
    
    # Check for deleted/removed
    if '[deleted]' in text or '[removed]' in text:
        return False
    
    # Check for links-only comments
    if text.count('http') > 2 and word_count < 100:
        return False
    
    # If multiplist pattern filtering is enabled, check for indicators
    if require_multiplist_patterns:
        if not has_multiplist_indicators(text):
            return False
    
    # If all filters pass, the entry is suitable
    return True


def extract_multiplist_candidates(ds, dataset_name='relationship_advice', max_samples=10000, 
                                  progress=True, require_multiplist_patterns=False):
    """
    Extract comments that are candidates for multiplist labeling.
    
    Args:
        ds: Dataset to extract from
        dataset_name: Name of the dataset/subreddit (used for sample IDs and metadata)
        max_samples: Maximum number of candidates to extract
        progress: Whether to show progress bar
        require_multiplist_patterns: If True, only include entries with multiplist indicators
    
    Returns a list of dictionaries with sample data and multiplist scores.
    """
    candidates = []
    filter_stats = Counter()
    
    # Create a short prefix for sample IDs (first 2-3 letters of dataset name)
    prefix = dataset_name[:3].lower() if len(dataset_name) >= 3 else dataset_name.lower()
    
    # Sample or iterate through dataset
    n_to_check = min(max_samples * 10, len(ds))  # Check more than we need
    indices = random.sample(range(len(ds)), n_to_check) if n_to_check < len(ds) else range(len(ds))
    
    iterator = tqdm(indices, desc="Scanning for multiplist candidates") if progress else indices
    
    for idx in iterator:
        entry = ds[idx]
        
        is_suitable = filter_for_labeling(entry, require_multiplist_patterns=require_multiplist_patterns)
        
        if is_suitable and len(candidates) < max_samples:
            text = get_text_field(entry)
            
            candidates.append({
                'sample_id': f"{prefix}_{idx}",
                'dataset': dataset_name,
                'sample_type': 'reddit_comment',
                'word_count': len(text.split()),
                'text': text,
            })
        
        if len(candidates) >= max_samples:
            break
    
    print(f"\n" + "="*60)
    print("FILTERING RESULTS")
    print("="*60)
    print(f"\nTotal comments scanned: {sum(filter_stats.values())}")
    print(f"Multiplist candidates found: {len(candidates)}")
    print(f"\nFilter breakdown:")
    for reason, count in filter_stats.most_common():
        pct = 100 * count / sum(filter_stats.values())
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    return candidates, filter_stats


# ============================================================================
# STEP 4: Create stratified sample for labeling
# ============================================================================

def create_labeling_sample(candidates, n_samples=1000):
    """
    Create a stratified sample prioritizing medium-length comments.
    
    We want diversity in:
    - Comment length (prioritize 200-500 words - good for substantive reasoning)
    - Topic variety (implicit in random sampling)
    """
    if not candidates:
        print("No candidates to sample from!")
        return []
    
    def quality_score(candidate):
        """
        Score candidates based on word count, prioritizing medium-length comments.
        Ideal range: 200-500 words (substantive but not too long).
        Returns a score where higher is better.
        """
        wc = candidate['word_count']
        # Ideal word count range
        ideal_min, ideal_max = 200, 500
        
        if ideal_min <= wc <= ideal_max:
            # In ideal range - score based on how close to center
            center = (ideal_min + ideal_max) / 2
            distance_from_center = abs(wc - center)
            max_distance = (ideal_max - ideal_min) / 2
            # Score: 100 at center, decreasing to 50 at edges
            return 100 - (distance_from_center / max_distance) * 50
        elif wc < ideal_min:
            # Below ideal - score decreases with distance
            return max(0, 50 - (ideal_min - wc) * 0.5)
        else:
            # Above ideal - score decreases with distance (but less penalty)
            return max(0, 50 - (wc - ideal_max) * 0.1)
    
    # Sort by quality score (descending)
    sorted_candidates = sorted(candidates, key=quality_score, reverse=True)
    
    # Take top candidates plus some random selection for diversity
    n_top = min(n_samples // 2, len(sorted_candidates))
    n_random = min(n_samples - n_top, len(sorted_candidates) - n_top)
    
    top_samples = sorted_candidates[:n_top]
    remaining = sorted_candidates[n_top:]
    random_samples = random.sample(remaining, n_random) if remaining else []
    
    sample = top_samples + random_samples
    random.shuffle(sample)
    
    print(f"\n" + "="*60)
    print("LABELING SAMPLE CREATED")
    print("="*60)
    print(f"\nTotal samples: {len(sample)}")
    print(f"  Top quality: {n_top}")
    print(f"  Random selection: {n_random}")
    
    # Word count distribution
    word_counts = [s['word_count'] for s in sample]
    print(f"\nWord count distribution:")
    print(f"  Mean: {sum(word_counts)/len(word_counts):.0f}")
    print(f"  Median: {sorted(word_counts)[len(word_counts)//2]:.0f}")
    print(f"  Range: {min(word_counts)} - {max(word_counts)}")
    
    # Count by length category
    ideal_range = sum(1 for wc in word_counts if 200 <= wc <= 500)
    short = sum(1 for wc in word_counts if wc < 200)
    long = sum(1 for wc in word_counts if wc > 500)
    print(f"\nLength categories:")
    print(f"  Short (<200 words): {short} ({100*short/len(sample):.1f}%)")
    print(f"  Ideal (200-500 words): {ideal_range} ({100*ideal_range/len(sample):.1f}%)")
    print(f"  Long (>500 words): {long} ({100*long/len(sample):.1f}%)")
    
    return sample


def save_labeling_sample(sample, dataset_name='relationship_advice', output_path=None):
    """
    Save the sample to CSV for labeling.
    
    Args:
        sample: List of sample dictionaries
        dataset_name: Name of the dataset (used for default filename if output_path not provided)
        output_path: Path to save CSV file (defaults to {dataset_name}_labeling_sample.csv)
    """
    if output_path is None:
        output_path = f"{dataset_name}_labeling_sample.csv"
    df = pd.DataFrame(sample)
    
    # Reorder columns (only include columns that exist)
    priority_cols = ['sample_id', 'dataset', 'sample_type', 'word_count', 'text']
    other_cols = [c for c in df.columns if c not in priority_cols]
    column_order = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[column_order]
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} samples to {output_path}")
    
    return df


# ============================================================================
# STEP 5: Preview examples
# ============================================================================

# def preview_examples(sample, n_examples=5):
#     """Preview examples by multiplist score tier."""
    
#     print("\n" + "="*60)
#     print("EXAMPLE SAMPLES FOR REVIEW")
#     print("="*60)
    
#     # Sort by score
#     sorted_sample = sorted(sample, key=lambda x: x['multiplist_score'], reverse=True)
    
#     # High scorers
#     print("\n--- HIGH MULTIPLIST SCORE (top examples) ---")
#     for i, s in enumerate(sorted_sample[:n_examples]):
#         print(f"\n[Example {i+1}] Score: {s['multiplist_score']} | {s['word_count']} words")
#         print(f"Strong matches: {s['strong_matches']}")
#         print(f"Moderate matches: {s['moderate_matches']}")
#         print("-" * 50)
#         preview = s['text'][:500] + "..." if len(s['text']) > 500 else s['text']
#         print(preview)
    
#     # Medium scorers
#     mid_start = len(sorted_sample) // 2
#     print("\n--- MEDIUM MULTIPLIST SCORE (middle examples) ---")
#     for i, s in enumerate(sorted_sample[mid_start:mid_start+3]):
#         print(f"\n[Example {i+1}] Score: {s['multiplist_score']} | {s['word_count']} words")
#         print(f"Strong matches: {s['strong_matches']}")
#         print("-" * 50)
#         preview = s['text'][:400] + "..." if len(s['text']) > 400 else s['text']
#         print(preview)


# ============================================================================
# STEP 6: Analyze potential distribution
# ============================================================================

# def analyze_sample_characteristics(sample):
#     """Analyze characteristics of the sample."""
    
#     print("\n" + "="*60)
#     print("SAMPLE CHARACTERISTICS")
#     print("="*60)
    
#     # Pattern frequency
#     all_strong = []
#     all_moderate = []
#     for s in sample:
#         all_strong.extend(json.loads(s['strong_matches']))
#         all_moderate.extend(json.loads(s['moderate_matches']))
    
#     print("\nMost common strong multiplist patterns:")
#     for pattern, count in Counter(all_strong).most_common(10):
#         print(f"  '{pattern}': {count}")
    
#     print("\nMost common moderate multiplist patterns:")
#     for pattern, count in Counter(all_moderate).most_common(10):
#         print(f"  '{pattern}': {count}")
    
#     print("\n⚠️  NOTE: These are heuristically-selected candidates.")
#     print("    Actual labeling will determine true epistemic stance.")
#     print("    Some may turn out to be evaluativist (reasoning through options)")
#     print("    rather than true multiplist (refusing to evaluate).")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(split_name='relationship_advice', require_multiplist_patterns=False):
    """
    Main workflow for multiplist extraction from a Reddit subreddit split.
    
    Args:
        split_name: Name of the subreddit split to process (default: 'relationship_advice')
        require_multiplist_patterns: If True, only include entries with multiplist indicators
    """
    
    # Set random seed
    random.seed(42)
    
    print("="*60)
    print(f"{split_name.upper()} - MULTIPLIST CANDIDATE EXTRACTION")
    print("="*60)
    
    if require_multiplist_patterns:
        print("\n⚠️  Multiplist pattern filtering ENABLED")
        print("    Only samples with multiplist indicators will be included.")
    else:
        print("\nℹ️  Multiplist pattern filtering DISABLED")
        print("    All samples passing basic filters will be included.")
    
    # Load dataset
    ds = load_reddit_split(split_name)
    
    # Explore structure
    # explore_structure(ds, n_samples=3)
    
    # Extract candidates
    candidates, filter_stats = extract_multiplist_candidates(
        ds, 
        dataset_name=split_name,
        max_samples=3000,  # Get more than we need for selection
        require_multiplist_patterns=require_multiplist_patterns
    )
    
    if not candidates:
        print("\n❌ No multiplist candidates found!")
        print("The dataset structure may be different than expected.")
        print("Check the column names and adjust get_text_field() if needed.")
        return
    
    # Create labeling sample
    sample = create_labeling_sample(candidates, n_samples=1000)
    
    # Save sample
    df = save_labeling_sample(sample, dataset_name=split_name)
    
    # # Preview examples
    # preview_examples(sample, n_examples=5)
    
    # # Analyze characteristics
    # analyze_sample_characteristics(sample)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"""
    1. Review {split_name}_labeling_sample.csv
    2. Run pilot labeling on 50-100 examples using label_multiplist_stance.py
    3. Validate that these are actually multiplist (not soft evaluativist)
    4. Combine with:
       - PERSUADE essays (absolutist)
       - CMV delta responses (evaluativist)
    5. Train balanced classifier
    
    Key validation questions:
    - Do "only you can decide" comments truly refuse to evaluate?
    - Or do they provide reasoning and then defer ("I think X, but only you can decide")?
    - The latter would be evaluativist, not multiplist.
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract samples from Reddit subreddit dataset for epistemic stance labeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explore_reddit_ds.py relationship_advice
  python explore_reddit_ds.py changemyview --filter-multiplist
  python explore_reddit_ds.py AskHistorians --filter-multiplist
        """
    )
    parser.add_argument(
        'split_name',
        type=str,
        nargs='?',
        default='relationship_advice',
        help='Name of the subreddit split to process (default: relationship_advice)'
    )
    parser.add_argument(
        '--filter-multiplist',
        action='store_true',
        help='Only include samples that contain multiplist linguistic indicators'
    )
    
    args = parser.parse_args()
    
    main(args.split_name, require_multiplist_patterns=args.filter_multiplist)
