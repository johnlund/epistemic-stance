"""
ChangeMyView Dataset Exploration and Preparation for Epistemic Stance Labeling
===============================================================================

This script downloads and analyzes the ChangeMyView dataset to prepare a subset
for epistemic stance labeling.

ChangeMyView (CMV) is ideal for epistemic stance classification because:
- Users explicitly state positions and reasoning
- Responders must engage with arguments substantively
- Delta (Δ) awards indicate successful persuasion
- Contains natural examples of all three epistemic stances

Requirements:
    pip install convokit datasets pandas tqdm

Usage:
    python explore_cmv.py
"""

import pandas as pd
from collections import Counter
import json
import random
from tqdm import tqdm
import re

# ============================================================================
# STEP 1: Load the ChangeMyView dataset
# ============================================================================

def load_cmv_convokit():
    """
    Load CMV using Cornell ConvoKit (recommended - has persuasion labels).
    
    This includes the "Winning Arguments" corpus with delta labels.
    """
    from convokit import Corpus, download
    
    print("Downloading CMV corpus via ConvoKit...")
    corpus = Corpus(filename=download("winning-args-corpus"))
    
    print(f"Loaded corpus with:")
    print(f"  - {len(corpus.get_conversation_ids())} conversations")
    print(f"  - {len(corpus.get_utterance_ids())} utterances")
    print(f"  - {len(corpus.get_speaker_ids())} speakers")
    
    return corpus


def load_cmv_huggingface():
    """
    Load CMV using HuggingFace datasets (ConcluGen version).
    
    This version has argument-conclusion pairs, good for understanding
    how arguments are structured.
    """
    from datasets import load_dataset
    
    print("Loading ConcluGen (CMV) from HuggingFace...")
    ds = load_dataset("webis/conclugen", "topics")
    
    print(f"Loaded {len(ds['train'])} argument-conclusion pairs")
    
    return ds


def load_cmv_combined():
    """
    Load both versions for comprehensive analysis.
    Returns dict with both datasets.
    """
    datasets = {}
    
    try:
        datasets['convokit'] = load_cmv_convokit()
    except Exception as e:
        print(f"Could not load ConvoKit version: {e}")
    
    try:
        datasets['huggingface'] = load_cmv_huggingface()
    except Exception as e:
        print(f"Could not load HuggingFace version: {e}")
    
    return datasets


# ============================================================================
# STEP 2: Explore dataset structure
# ============================================================================

def explore_convokit_structure(corpus, n_samples=5):
    """Examine the structure of the ConvoKit CMV corpus."""
    
    print("\n" + "="*60)
    print("CONVOKIT CMV STRUCTURE")
    print("="*60)
    
    # Get a sample conversation
    conv_ids = list(corpus.get_conversation_ids())[:n_samples]
    
    for conv_id in conv_ids[:2]:
        conv = corpus.get_conversation(conv_id)
        print(f"\n--- Conversation: {conv_id} ---")
        print(f"Metadata keys: {list(conv.meta.keys())}")
        
        # Get utterances in this conversation
        utterances = list(conv.iter_utterances())
        print(f"Number of utterances: {len(utterances)}")
        
        # Show first utterance (original post)
        if utterances:
            first = utterances[0]
            print(f"\nOriginal Post (first 300 chars):")
            print(f"  {first.text[:300]}...")
            print(f"  Speaker: {first.speaker.id}")
            print(f"  Metadata: {first.meta}")


def explore_huggingface_structure(ds, n_samples=5):
    """Examine the structure of the HuggingFace ConcluGen dataset."""
    
    print("\n" + "="*60)
    print("HUGGINGFACE CONCLUGEN STRUCTURE")
    print("="*60)
    
    print(f"\nDataset splits: {list(ds.keys())}")
    print(f"Train size: {len(ds['train'])}")
    print(f"Columns: {ds['train'].column_names}")
    
    print("\n--- Sample entries ---")
    for i in range(min(n_samples, len(ds['train']))):
        entry = ds['train'][i]
        print(f"\n[Sample {i+1}]")
        print(f"Argument (first 300 chars): {entry['argument'][:300]}...")
        print(f"Conclusion: {entry['conclusion']}")


def compute_statistics_convokit(corpus, sample_size=1000):
    """Compute statistics for ConvoKit corpus."""
    
    print("\n" + "="*60)
    print("CONVOKIT STATISTICS")
    print("="*60)
    
    conv_ids = list(corpus.get_conversation_ids())
    sample_ids = random.sample(conv_ids, min(sample_size, len(conv_ids)))
    
    utterance_lengths = []
    delta_awarded = 0
    op_lengths = []
    response_lengths = []
    
    for conv_id in tqdm(sample_ids, desc="Analyzing conversations"):
        conv = corpus.get_conversation(conv_id)
        utterances = list(conv.iter_utterances())
        
        for i, utt in enumerate(utterances):
            word_count = len(utt.text.split())
            utterance_lengths.append(word_count)
            
            if i == 0:
                op_lengths.append(word_count)
            else:
                response_lengths.append(word_count)
            
            # Check for delta in metadata
            if utt.meta.get('success', False):
                delta_awarded += 1
    
    print(f"\nBased on {len(sample_ids)} sampled conversations:")
    
    print(f"\nOriginal Post (OP) Statistics:")
    print(f"  Mean length: {sum(op_lengths)/len(op_lengths):.0f} words")
    print(f"  Median: {sorted(op_lengths)[len(op_lengths)//2]} words")
    print(f"  Range: {min(op_lengths)} - {max(op_lengths)} words")
    
    print(f"\nResponse Statistics:")
    print(f"  Mean length: {sum(response_lengths)/len(response_lengths):.0f} words")
    print(f"  Median: {sorted(response_lengths)[len(response_lengths)//2]} words")
    
    print(f"\nDelta (successful persuasion) rate: {delta_awarded} in sample")
    
    return {
        'op_lengths': op_lengths,
        'response_lengths': response_lengths,
        'utterance_lengths': utterance_lengths
    }


# ============================================================================
# STEP 3: Filter for high-quality epistemic reasoning samples
# ============================================================================

def has_reasoning_indicators(text):
    """
    Check if text contains indicators of epistemic reasoning.
    
    We want posts where people are actually reasoning about claims,
    not just stating preferences or making jokes.
    """
    reasoning_patterns = [
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
    
    text_lower = text.lower()
    matches = sum(1 for pattern in reasoning_patterns if re.search(pattern, text_lower))
    
    return matches >= 2  # At least 2 reasoning indicators


def filter_utterance_for_epistemic_labeling(text, min_words=50, max_words=None):
    """
    Filter a single utterance for suitability for epistemic stance labeling.
    
    Args:
        text: The utterance text
        min_words: Minimum word count (need enough content to assess stance)
        max_words: Maximum word count (for practical labeling)
    
    Returns: (is_suitable, reason)
    """
    word_count = len(text.split())
    
    # Length filters
    if word_count < min_words:
        return False, f"too_short ({word_count} words)"
    
    # No max limit for labeling stage
    # Long posts often contain the richest epistemic reasoning
    # if word_count > max_words:
    #     return False, f"too_long ({word_count} words)"
    
    # Check for deleted/removed content
    if '[deleted]' in text or '[removed]' in text:
        return False, "deleted_content"
    
    # Check for reasoning indicators
    if not has_reasoning_indicators(text):
        return False, "no_reasoning_indicators"
    
    # Filter out pure questions without argumentation
    sentences = text.split('.')
    question_ratio = sum(1 for s in sentences if '?' in s) / max(len(sentences), 1)
    if question_ratio > 0.7:
        return False, "mostly_questions"
    
    # Filter out very short sentences (likely not substantive)
    avg_sentence_length = word_count / max(len(sentences), 1)
    if avg_sentence_length < 8:
        return False, "choppy_writing"
    
    return True, "suitable"


def extract_suitable_samples_convokit(corpus, max_conversations=5000, 
                                       samples_per_conv=3):
    """
    Extract samples suitable for epistemic stance labeling from ConvoKit corpus.
    
    We extract:
    - Original posts (OP's stated position and reasoning)
    - Top-level responses (direct engagement with OP's argument)
    - Delta-awarded responses (successful persuasion - likely evaluativist)
    
    Returns a list of dictionaries with sample data and metadata.
    """
    suitable_samples = []
    filter_stats = Counter()
    
    conv_ids = list(corpus.get_conversation_ids())
    sample_conv_ids = random.sample(conv_ids, min(max_conversations, len(conv_ids)))
    
    for conv_id in tqdm(sample_conv_ids, desc="Extracting suitable samples"):
        conv = corpus.get_conversation(conv_id)
        utterances = list(conv.iter_utterances())
        
        samples_from_this_conv = 0
        
        for utt in utterances:
            if samples_from_this_conv >= samples_per_conv:
                break
            
            is_suitable, reason = filter_utterance_for_epistemic_labeling(utt.text)
            filter_stats[reason] += 1
            
            if is_suitable:
                # Determine if this is OP, response, or delta-awarded
                is_op = (utt.reply_to is None)
                is_delta = utt.meta.get('success', False)
                
                sample_type = 'original_post' if is_op else ('delta_response' if is_delta else 'response')
                
                suitable_samples.append({
                    'sample_id': f"{conv_id}_{utt.id}",
                    'conversation_id': conv_id,
                    'utterance_id': utt.id,
                    'sample_type': sample_type,
                    'is_delta_awarded': is_delta,
                    'speaker_id': utt.speaker.id,
                    'reply_to': utt.reply_to,
                    'word_count': len(utt.text.split()),
                    'text': utt.text,
                })
                samples_from_this_conv += 1
    
    print(f"\n" + "="*60)
    print("FILTERING RESULTS")
    print("="*60)
    print(f"\nTotal utterances processed: {sum(filter_stats.values())}")
    print(f"Suitable samples extracted: {filter_stats['suitable']}")
    print(f"\nFilter breakdown:")
    for reason, count in filter_stats.most_common():
        pct = 100 * count / sum(filter_stats.values())
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Distribution by type
    type_counts = Counter(s['sample_type'] for s in suitable_samples)
    print(f"\nSample type distribution:")
    for stype, count in type_counts.most_common():
        print(f"  {stype}: {count}")
    
    delta_count = sum(1 for s in suitable_samples if s['is_delta_awarded'])
    print(f"\nDelta-awarded samples: {delta_count}")
    
    return suitable_samples, filter_stats


def extract_suitable_samples_huggingface(ds, max_samples=5000):
    """
    Extract samples from HuggingFace ConcluGen dataset.
    
    Each entry has an argument and its conclusion, which is useful
    for understanding argument structure.
    """
    suitable_samples = []
    filter_stats = Counter()
    
    indices = random.sample(range(len(ds['train'])), min(max_samples, len(ds['train'])))
    
    for idx in tqdm(indices, desc="Extracting suitable samples"):
        entry = ds['train'][idx]
        argument = entry['argument']
        conclusion = entry['conclusion']
        
        is_suitable, reason = filter_utterance_for_epistemic_labeling(argument)
        filter_stats[reason] += 1
        
        if is_suitable:
            suitable_samples.append({
                'sample_id': entry['id'],
                'sample_type': 'argument_with_conclusion',
                'word_count': len(argument.split()),
                'text': argument,
                'conclusion': conclusion,
                'topic': entry.get('topic', ''),
            })
    
    print(f"\n" + "="*60)
    print("HUGGINGFACE FILTERING RESULTS")
    print("="*60)
    print(f"Total processed: {sum(filter_stats.values())}")
    print(f"Suitable: {filter_stats['suitable']}")
    
    return suitable_samples, filter_stats


# ============================================================================
# STEP 4: Create stratified sample for labeling
# ============================================================================

def create_labeling_sample(suitable_samples, n_samples=3000, 
                           prioritize_delta=True, balance_types=True):
    """
    Create a stratified sample for epistemic stance labeling.
    
    Stratification goals:
    - Include delta-awarded responses (likely evaluativist - they convinced someone)
    - Include original posts (OP's stated reasoning)
    - Include regular responses (may show various stances)
    - Ensure diversity of word counts
    """
    
    if not suitable_samples:
        print("No suitable samples to create labeling set from!")
        return []
    
    # Separate by type
    ops = [s for s in suitable_samples if s.get('sample_type') == 'original_post']
    deltas = [s for s in suitable_samples if s.get('is_delta_awarded', False)]
    responses = [s for s in suitable_samples 
                 if s.get('sample_type') == 'response' and not s.get('is_delta_awarded', False)]
    other = [s for s in suitable_samples 
             if s.get('sample_type') not in ['original_post', 'response', 'delta_response']]
    
    print(f"\nAvailable samples by type:")
    print(f"  Original posts: {len(ops)}")
    print(f"  Delta-awarded responses: {len(deltas)}")
    print(f"  Regular responses: {len(responses)}")
    print(f"  Other (e.g., HuggingFace arguments): {len(other)}")
    
    sample = []
    
    if balance_types and ops and responses:
        # Balanced sampling across types
        n_per_type = n_samples // 3
        
        # Prioritize delta responses (oversample)
        n_delta = min(len(deltas), n_per_type + n_per_type // 2) if prioritize_delta else min(len(deltas), n_per_type)
        n_ops = min(len(ops), n_per_type)
        n_responses = n_samples - n_delta - n_ops
        
        sample.extend(random.sample(deltas, n_delta) if len(deltas) >= n_delta else deltas)
        sample.extend(random.sample(ops, n_ops) if len(ops) >= n_ops else ops)
        sample.extend(random.sample(responses, min(n_responses, len(responses))))
        
    else:
        # Simple random sampling
        all_samples = ops + deltas + responses + other
        sample = random.sample(all_samples, min(n_samples, len(all_samples)))
    
    random.shuffle(sample)
    
    print(f"\n" + "="*60)
    print("LABELING SAMPLE CREATED")
    print("="*60)
    print(f"\nTotal samples: {len(sample)}")
    
    type_counts = Counter(s.get('sample_type', 'unknown') for s in sample)
    print(f"\nType distribution in sample:")
    for stype, count in type_counts.most_common():
        print(f"  {stype}: {count} ({100*count/len(sample):.1f}%)")
    
    delta_count = sum(1 for s in sample if s.get('is_delta_awarded', False))
    print(f"\nDelta-awarded in sample: {delta_count} ({100*delta_count/len(sample):.1f}%)")
    
    word_counts = [s['word_count'] for s in sample]
    print(f"\nWord count distribution:")
    print(f"  Mean: {sum(word_counts)/len(word_counts):.0f}")
    print(f"  Median: {sorted(word_counts)[len(word_counts)//2]}")
    print(f"  Range: {min(word_counts)} - {max(word_counts)}")
    
    return sample


def save_labeling_sample(sample, output_path="cmv_labeling_sample.csv"):
    """Save the sample to CSV for labeling."""
    df = pd.DataFrame(sample)
    
    # Reorder columns for clarity
    priority_cols = ['sample_id', 'conversation_id', 'sample_type', 
                     'is_delta_awarded', 'word_count', 'text']
    other_cols = [c for c in df.columns if c not in priority_cols]
    column_order = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[column_order]
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} samples to {output_path}")
    
    return df


# ============================================================================
# STEP 5: Preview examples
# ============================================================================

def preview_examples(sample, n_examples=3):
    """
    Preview example samples to understand what we're labeling.
    """
    print("\n" + "="*60)
    print("EXAMPLE SAMPLES FOR REVIEW")
    print("="*60)
    
    # Show examples by type
    for sample_type in ['original_post', 'delta_response', 'response']:
        type_samples = [s for s in sample if s.get('sample_type') == sample_type]
        if type_samples:
            print(f"\n--- {sample_type.upper()} EXAMPLES ---")
            for i, s in enumerate(random.sample(type_samples, min(n_examples, len(type_samples)))):
                print(f"\n[Example {i+1}] (ID: {s['sample_id']}, {s['word_count']} words)")
                if s.get('is_delta_awarded'):
                    print("  *** DELTA AWARDED - This argument changed OP's mind ***")
                print("-" * 50)
                # Show first 600 chars
                preview = s['text'][:600] + "..." if len(s['text']) > 600 else s['text']
                print(preview)


# ============================================================================
# STEP 6: Analyze potential epistemic stance distribution
# ============================================================================

def analyze_potential_stances(sample):
    """
    Do a rough heuristic analysis to estimate stance distribution.
    
    This helps us understand what to expect before labeling.
    """
    print("\n" + "="*60)
    print("PRELIMINARY STANCE INDICATORS (HEURISTIC)")
    print("="*60)
    
    absolutist_indicators = [
        r'\bobviously\b', r'\bclearly\b', r'\bundeniably\b',
        r'\bthe fact is\b', r'\bthe truth is\b', r'\bno doubt\b',
        r'\bwithout question\b', r'\babsolutely\b', r'\bdefinitely\b',
        r'\beveryone knows\b', r'\bit\'s clear that\b',
    ]
    
    multiplist_indicators = [
        r'\bjust my opinion\b', r'\beveryone.+entitled\b', 
        r'\bwho\'s to say\b', r'\bit\'s subjective\b',
        r'\bdepends on the person\b', r'\bboth.+valid\b',
        r'\bneither.+wrong\b', r'\bnot for me to judge\b',
    ]
    
    evaluativist_indicators = [
        r'\bthe evidence suggests\b', r'\bon balance\b',
        r'\bwhile.+however\b', r'\balthough.+still\b',
        r'\bi could be wrong\b', r'\bmore likely\b',
        r'\bstronger argument\b', r'\bbetter supported\b',
        r'\bhaving considered\b', r'\bweighing\b',
        r'\bI\'ve changed my mind\b', r'\byou\'ve convinced me\b',
    ]
    
    stance_counts = {'absolutist_signals': 0, 'multiplist_signals': 0, 
                     'evaluativist_signals': 0, 'unclear': 0}
    
    for s in sample:
        text_lower = s['text'].lower()
        
        abs_matches = sum(1 for p in absolutist_indicators if re.search(p, text_lower))
        mult_matches = sum(1 for p in multiplist_indicators if re.search(p, text_lower))
        eval_matches = sum(1 for p in evaluativist_indicators if re.search(p, text_lower))
        
        # Delta-awarded responses are likely evaluativist
        if s.get('is_delta_awarded'):
            eval_matches += 2
        
        if eval_matches > abs_matches and eval_matches > mult_matches:
            stance_counts['evaluativist_signals'] += 1
        elif abs_matches > mult_matches:
            stance_counts['absolutist_signals'] += 1
        elif mult_matches > 0:
            stance_counts['multiplist_signals'] += 1
        else:
            stance_counts['unclear'] += 1
    
    print("\nHeuristic stance signal distribution:")
    for stance, count in stance_counts.items():
        print(f"  {stance}: {count} ({100*count/len(sample):.1f}%)")
    
    print("\n⚠️  Note: This is a rough heuristic, not actual labels!")
    print("    The actual labeling will be more nuanced.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main workflow for CMV dataset exploration."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("="*60)
    print("CHANGEMYVIEW DATASET EXPLORATION")
    print("="*60)
    
    # Try to load ConvoKit version first (has delta labels)
    corpus = None
    try:
        corpus = load_cmv_convokit()
    except Exception as e:
        print(f"Could not load ConvoKit: {e}")
        print("Trying HuggingFace version...")
    
    if corpus:
        # ConvoKit workflow
        explore_convokit_structure(corpus)
        stats = compute_statistics_convokit(corpus, sample_size=2000)
        suitable_samples, filter_stats = extract_suitable_samples_convokit(
            corpus, max_conversations=5000
        )
    else:
        # Fallback to HuggingFace
        ds = load_cmv_huggingface()
        explore_huggingface_structure(ds)
        suitable_samples, filter_stats = extract_suitable_samples_huggingface(
            ds, max_samples=5000
        )
    
    # Create labeling sample
    sample = create_labeling_sample(suitable_samples, n_samples=3000)
    
    # Save sample
    df = save_labeling_sample(sample, "cmv_labeling_sample.csv")
    
    # Preview examples
    preview_examples(sample, n_examples=3)
    
    # Analyze potential stance distribution
    analyze_potential_stances(sample)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
    1. Review cmv_labeling_sample.csv
    2. Run pilot labeling on 50-100 examples
    3. Validate labeling approach
    4. Scale to full 3000 samples
    5. Train classifier
    
    Key advantages of CMV data:
    - Delta-awarded responses are likely evaluativist (they convinced someone!)
    - Original posts show how people frame their initial positions
    - Responses show engagement with arguments
    - Natural language, not formal essays
    """)


if __name__ == "__main__":
    main()
