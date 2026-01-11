"""
Relationship Advice Reddit Data Exploration and Preparation
============================================================

This script downloads and filters the relationship_advice subreddit data
from HuggingFaceGECLM/REDDIT_comments to find multiplist reasoning examples.

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
    python explore_relationship_advice.py
"""

import pandas as pd
from collections import Counter
import json
import random
from tqdm import tqdm
import re

# ============================================================================
# STEP 1: Load the relationship_advice split
# ============================================================================

def load_relationship_advice():
    """
    Load the relationship_advice split from HuggingFaceGECLM/REDDIT_comments.
    
    This dataset contains Reddit comments from 50 high-quality subreddits,
    including relationship_advice.
    """
    from datasets import load_dataset
    
    print("Loading dataset from HuggingFaceGECLM/REDDIT_comments...")
    print("(This may take a few minutes on first download)")
    
    # Load the default config (the dataset doesn't have separate configs per subreddit)
    ds = load_dataset(
        "HuggingFaceGECLM/REDDIT_comments",
        "default"
    )
    
    print(f"\nLoaded dataset with splits: {list(ds.keys())}")
    
    # Get the main split (usually 'train')
    main_split = list(ds.keys())[0]
    print(f"Using split: {main_split}")
    print(f"Number of comments: {len(ds[main_split])}")
    print(f"Columns: {ds[main_split].column_names}")
    
    # Filter for relationship_advice subreddit
    # Check what field contains the subreddit name
    sample_entry = ds[main_split][0]
    print(f"\nSample entry keys: {list(sample_entry.keys())}")
    
    # Try to find the subreddit field (common names: 'subreddit', 'subreddit_name', 'subredditName')
    subreddit_field = None
    for field in ['subreddit', 'subreddit_name', 'subredditName', 'subredditNamePrefixed']:
        if field in sample_entry:
            subreddit_field = field
            print(f"Found subreddit field: {field}")
            print(f"Sample value: {sample_entry[field]}")
            break
    
    if not subreddit_field:
        print("\n⚠️  Warning: Could not find subreddit field. Available fields:")
        for key, value in sample_entry.items():
            print(f"  {key}: {type(value).__name__}")
        print("\nReturning full dataset - you may need to filter manually.")
        return ds[main_split]
    
    # Filter for relationship_advice
    print(f"\nFiltering for 'relationship_advice' subreddit...")
    filtered_ds = ds[main_split].filter(
        lambda x: 'relationship_advice' in str(x.get(subreddit_field, '')).lower()
    )
    
    print(f"Filtered to {len(filtered_ds)} relationship_advice comments")
    print(f"(from {len(ds[main_split])} total comments)")
    
    return filtered_ds

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
]

# Anti-patterns: These suggest evaluativist or absolutist, NOT multiplist
ANTI_MULTIPLIST_PATTERNS = [
    # Evaluativist markers (weighing evidence)
    r'\bthe (better|best|stronger) (option|choice|argument)\b',
    r'\bon balance\b',
    r'\bweighing\b',
    r'\bmore (likely|reasonable|justified)\b',
    r'\bthe evidence (suggests|shows)\b',
    
    # Absolutist markers (certainty)
    r'\byou (should|must|need to) (definitely|absolutely)\b',
    r'\bthere\'s no (question|doubt)\b',
    r'\bobviously\b',
    r'\bclearly (you should|the answer)\b',
    r'\b(dump|leave|divorce) (him|her|them)\b',  # Strong directive advice
    r'\bred flag\b',  # Definitive judgment
    r'\bdeal ?breaker\b',
]


def score_multiplist_indicators(text):
    """
    Score a text for multiplist indicators.
    
    Returns:
        dict with:
        - strong_matches: list of strong pattern matches
        - moderate_matches: list of moderate pattern matches
        - anti_matches: list of anti-multiplist pattern matches
        - multiplist_score: overall score (higher = more multiplist)
    """
    text_lower = text.lower()
    
    strong_matches = []
    for pattern in MULTIPLIST_STRONG_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            strong_matches.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
    
    moderate_matches = []
    for pattern in MULTIPLIST_MODERATE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            moderate_matches.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
    
    anti_matches = []
    for pattern in ANTI_MULTIPLIST_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            anti_matches.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
    
    # Calculate score: strong patterns worth more, anti-patterns subtract
    score = (len(strong_matches) * 3) + (len(moderate_matches) * 1) - (len(anti_matches) * 2)
    
    return {
        'strong_matches': strong_matches,
        'moderate_matches': moderate_matches,
        'anti_matches': anti_matches,
        'multiplist_score': score,
        'n_strong': len(strong_matches),
        'n_moderate': len(moderate_matches),
        'n_anti': len(anti_matches),
    }


# ============================================================================
# STEP 3: Filter for suitable samples
# ============================================================================

def get_text_field(entry):
    """Extract the text content from an entry, handling different possible field names."""
    # Try common field names
    for field in ['normalizedBody']:
        if field in entry and entry[field]:
            return entry[field]
    return None


def filter_for_multiplist_labeling(entry, min_words=50, max_words=None):
    """
    Filter a single entry for suitability for multiplist labeling.
    
    We want comments that:
    - Are substantive (not too short)
    - Contain reasoning (not just "lol" or "this")
    - Show multiplist indicators
    - Don't have strong absolutist/evaluativist markers
    
    Returns: (is_suitable, reason, multiplist_score_data)
    """
    text = get_text_field(entry)
    
    if not text:
        return False, "no_text", None
    
    # Clean up text
    text = text.strip()
    
    # Basic filters
    word_count = len(text.split())
    
    if word_count < min_words:
        return False, f"too_short ({word_count} words)", None
    
    # if word_count > max_words:
    #     return False, f"too_long ({word_count} words)", None
    
    # Check for deleted/removed
    if '[deleted]' in text or '[removed]' in text:
        return False, "deleted_content", None
    
    # Check for links-only comments
    if text.count('http') > 2 and word_count < 100:
        return False, "mostly_links", None
    
    # Score for multiplist indicators
    score_data = score_multiplist_indicators(text)
    
    # We want comments with some multiplist signal
    # At least one strong match, OR multiple moderate matches
    if score_data['n_strong'] >= 1:
        return True, "strong_multiplist", score_data
    elif score_data['n_moderate'] >= 2 and score_data['n_anti'] == 0:
        return True, "moderate_multiplist", score_data
    elif score_data['multiplist_score'] >= 2:
        return True, "positive_score", score_data
    else:
        return False, "no_multiplist_signal", score_data


def extract_multiplist_candidates(ds, max_samples=10000, progress=True):
    """
    Extract comments that are candidates for multiplist labeling.
    
    Returns a list of dictionaries with sample data and multiplist scores.
    """
    candidates = []
    filter_stats = Counter()
    
    # Sample or iterate through dataset
    n_to_check = min(max_samples * 10, len(ds))  # Check more than we need
    indices = random.sample(range(len(ds)), n_to_check) if n_to_check < len(ds) else range(len(ds))
    
    iterator = tqdm(indices, desc="Scanning for multiplist candidates") if progress else indices
    
    for idx in iterator:
        entry = ds[idx]
        
        is_suitable, reason, score_data = filter_for_multiplist_labeling(entry)
        filter_stats[reason] += 1
        
        if is_suitable and len(candidates) < max_samples:
            text = get_text_field(entry)
            
            candidates.append({
                'sample_id': f"ra_{idx}",
                'dataset': 'relationship_advice',
                'sample_type': 'reddit_comment',
                'word_count': len(text.split()),
                'text': text,
                'filter_reason': reason,
                'multiplist_score': score_data['multiplist_score'],
                'n_strong_matches': score_data['n_strong'],
                'n_moderate_matches': score_data['n_moderate'],
                'n_anti_matches': score_data['n_anti'],
                'strong_matches': json.dumps(score_data['strong_matches'][:5]),  # Limit for CSV
                'moderate_matches': json.dumps(score_data['moderate_matches'][:5]),
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
    Create a stratified sample prioritizing high multiplist scores.
    
    We want diversity in:
    - Multiplist signal strength
    - Comment length
    - Topic variety (implicit in random sampling)
    """
    if not candidates:
        print("No candidates to sample from!")
        return []
    
    # Sort by multiplist score (descending)
    sorted_candidates = sorted(candidates, key=lambda x: x['multiplist_score'], reverse=True)
    
    # Take top scorers plus some random selection
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
    print(f"  Top scorers: {n_top}")
    print(f"  Random selection: {n_random}")
    
    # Score distribution
    scores = [s['multiplist_score'] for s in sample]
    print(f"\nMultiplist score distribution:")
    print(f"  Mean: {sum(scores)/len(scores):.1f}")
    print(f"  Range: {min(scores)} - {max(scores)}")
    
    # Word count distribution
    word_counts = [s['word_count'] for s in sample]
    print(f"\nWord count distribution:")
    print(f"  Mean: {sum(word_counts)/len(word_counts):.0f}")
    print(f"  Range: {min(word_counts)} - {max(word_counts)}")
    
    # Strong vs moderate
    n_strong = sum(1 for s in sample if s['n_strong_matches'] > 0)
    print(f"\nSamples with strong multiplist patterns: {n_strong} ({100*n_strong/len(sample):.1f}%)")
    
    return sample


def save_labeling_sample(sample, output_path="relationship_advice_labeling_sample.csv"):
    """Save the sample to CSV for labeling."""
    df = pd.DataFrame(sample)
    
    # Reorder columns
    priority_cols = ['sample_id', 'dataset', 'sample_type', 'word_count', 
                     'multiplist_score', 'n_strong_matches', 'n_moderate_matches',
                     'filter_reason', 'text']
    other_cols = [c for c in df.columns if c not in priority_cols]
    column_order = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[column_order]
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} samples to {output_path}")
    
    return df


# ============================================================================
# STEP 5: Preview examples
# ============================================================================

def preview_examples(sample, n_examples=5):
    """Preview examples by multiplist score tier."""
    
    print("\n" + "="*60)
    print("EXAMPLE SAMPLES FOR REVIEW")
    print("="*60)
    
    # Sort by score
    sorted_sample = sorted(sample, key=lambda x: x['multiplist_score'], reverse=True)
    
    # High scorers
    print("\n--- HIGH MULTIPLIST SCORE (top examples) ---")
    for i, s in enumerate(sorted_sample[:n_examples]):
        print(f"\n[Example {i+1}] Score: {s['multiplist_score']} | {s['word_count']} words")
        print(f"Strong matches: {s['strong_matches']}")
        print(f"Moderate matches: {s['moderate_matches']}")
        print("-" * 50)
        preview = s['text'][:500] + "..." if len(s['text']) > 500 else s['text']
        print(preview)
    
    # Medium scorers
    mid_start = len(sorted_sample) // 2
    print("\n--- MEDIUM MULTIPLIST SCORE (middle examples) ---")
    for i, s in enumerate(sorted_sample[mid_start:mid_start+3]):
        print(f"\n[Example {i+1}] Score: {s['multiplist_score']} | {s['word_count']} words")
        print(f"Strong matches: {s['strong_matches']}")
        print("-" * 50)
        preview = s['text'][:400] + "..." if len(s['text']) > 400 else s['text']
        print(preview)


# ============================================================================
# STEP 6: Analyze potential distribution
# ============================================================================

def analyze_sample_characteristics(sample):
    """Analyze characteristics of the sample."""
    
    print("\n" + "="*60)
    print("SAMPLE CHARACTERISTICS")
    print("="*60)
    
    # Pattern frequency
    all_strong = []
    all_moderate = []
    for s in sample:
        all_strong.extend(json.loads(s['strong_matches']))
        all_moderate.extend(json.loads(s['moderate_matches']))
    
    print("\nMost common strong multiplist patterns:")
    for pattern, count in Counter(all_strong).most_common(10):
        print(f"  '{pattern}': {count}")
    
    print("\nMost common moderate multiplist patterns:")
    for pattern, count in Counter(all_moderate).most_common(10):
        print(f"  '{pattern}': {count}")
    
    print("\n⚠️  NOTE: These are heuristically-selected candidates.")
    print("    Actual labeling will determine true epistemic stance.")
    print("    Some may turn out to be evaluativist (reasoning through options)")
    print("    rather than true multiplist (refusing to evaluate).")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main workflow for relationship_advice multiplist extraction."""
    
    # Set random seed
    random.seed(42)
    
    print("="*60)
    print("RELATIONSHIP ADVICE - MULTIPLIST CANDIDATE EXTRACTION")
    print("="*60)
    
    # Load dataset
    ds = load_relationship_advice()
    
    # Explore structure
    # explore_structure(ds, n_samples=3)
    
    # # Extract candidates
    # candidates, filter_stats = extract_multiplist_candidates(
    #     ds, 
    #     max_samples=3000  # Get more than we need for selection
    # )
    
    # if not candidates:
    #     print("\n❌ No multiplist candidates found!")
    #     print("The dataset structure may be different than expected.")
    #     print("Check the column names and adjust get_text_field() if needed.")
    #     return
    
    # # Create labeling sample
    # sample = create_labeling_sample(candidates, n_samples=1000)
    
    # # Save sample
    # df = save_labeling_sample(sample, "relationship_advice_labeling_sample.csv")
    
    # # Preview examples
    # preview_examples(sample, n_examples=5)
    
    # # Analyze characteristics
    # analyze_sample_characteristics(sample)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
    1. Review relationship_advice_labeling_sample.csv
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
    main()
