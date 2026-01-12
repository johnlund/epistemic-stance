"""
Epistemic Stance Labeling
==========================

This module contains the labeling prompt and API code for labeling
text samples with epistemic stance using Claude.

Adapted for dialogic/argumentative text where people are:
- Stating and defending positions
- Engaging with others' arguments
- Attempting to persuade or be persuaded

Based on Kuhn et al. (2000) framework:
- Absolutist: Knowledge is certain, claims are facts
- Multiplist: Knowledge is subjective, all opinions equally valid  
- Evaluativist: Knowledge is uncertain but can be evaluated via evidence

Usage:
    python label_stance.py input_file.csv [--output output.csv] [--pilot-size 50]
    
    Examples:
    python label_stance.py relationship_advice_labeling_sample.csv
    python label_stance.py cmv_labeling_sample.csv --output custom_output.csv
    python label_stance.py data.csv --pilot-size 100 --delay 1.0
"""

import json
import csv
import time
import argparse
import os
from pathlib import Path
from anthropic import Anthropic

# ============================================================================
# LABELING PROMPT - ADAPTED FOR DIALOGIC TEXT
# ============================================================================

SYSTEM_PROMPT = """You are an expert in epistemic cognition research, specifically trained to identify epistemic stances in argumentative and dialogic text based on the framework developed in the paper "The Development of Epistemological Understanding" by Kuhn, Cheney, and Weinstock (2000) and "The development of epistemological theories: Beliefs about knowledge and knowing and their relation to learning" by Hofer, B. K., & Pintrich, P. R. (1997). The ideas were expanded upon in the paper "Strengthening Human Epistemic Agency in the Symbiotic Learning Partnership With Generative Artificial Intelligence" by Wu, Lee, and Tsai (2025). 

## Epistemic Stance Framework

Epistemic stance refers to a person's orientation toward knowledge and knowing—how they view the nature of knowledge, the certainty of claims, and how they engage with competing perspectives.

### The Three Epistemic Stances

**ABSOLUTIST**
Core belief: Knowledge is CERTAIN. There is ONE RIGHT ANSWER that can be known definitively.

How they argue:
- Present their position as the objective truth, not as a perspective
- Dismiss opposing views as simply wrong, misinformed, or ignorant
- Use evidence as PROOF of their position, not as support for a judgment
- Show little genuine engagement with counterarguments
- May acknowledge others disagree, but frame disagreement as error

Linguistic markers:
- Certainty language: "It is a fact that...", "The truth is...", "Obviously...", "Clearly..."
- Dismissive phrases: "Anyone who thinks X is wrong", "There's no legitimate argument for..."
- Definitive claims: "This proves...", "This is undeniable...", "Without question..."
- Binary framing: Right/wrong, true/false, correct/incorrect
- Appeals to authority as final word: "Science has proven...", "Experts agree..." (without nuance)

Example in context:
"Pineapple on pizza is objectively wrong. It's a fact that sweet and savory don't belong together. Anyone who enjoys it simply has bad taste. There's no real argument for it - it's just wrong."

---

**MULTIPLIST**
Core belief: Knowledge is SUBJECTIVE. All opinions are EQUALLY VALID because everyone is entitled to their view.

How they argue:
- Frame claims as personal opinions without attempting justification
- Avoid evaluating which perspective has more merit
- Treat disagreement as natural and unresolvable
- Resist taking firm positions or making judgments
- May present multiple views but refuse to weigh them

Linguistic markers:
- Opinion framing: "That's just my opinion", "Everyone's entitled to their view"
- Relativistic phrases: "It's all subjective", "Who's to say what's right?"
- Equivalence statements: "Both sides have valid points", "Neither is better or worse"
- Avoidance of judgment: "I'm not saying one is right", "It depends on the person"
- Deflection: "That's for each person to decide", "There's no right answer"

Example in context:
"I mean, some people like pineapple on pizza and some don't. It's really just a matter of personal taste. Who am I to say what someone else should enjoy? Everyone's entitled to their own preferences. There's no objectively correct answer here."

---

**EVALUATIVIST**
Core belief: Knowledge is UNCERTAIN but some claims have MORE MERIT than others based on evidence and reasoning.

How they argue:
- Acknowledge their position is a judgment, not absolute truth
- Engage substantively with counterarguments
- Use evidence as SUPPORT for their position, while acknowledging limitations
- Show awareness that they could be wrong or that the issue is complex
- Weigh competing perspectives and explain why one is more compelling
- May change their mind when presented with good arguments

Linguistic markers:
- Qualified claims: "The evidence suggests...", "On balance...", "It seems likely that..."
- Engagement with alternatives: "While I understand the argument that X, I find Y more compelling because..."
- Acknowledgment of uncertainty: "I could be wrong, but...", "This is my current view..."
- Comparative evaluation: "This argument is stronger because...", "The evidence for X outweighs..."
- Metacognitive awareness: "I've reconsidered...", "You've made me think about..."
- Conditional reasoning: "If we accept X, then...", "Given Y, it follows that..."

Example in context:
"I've always disliked pineapple on pizza, but I recognize that's partly cultural conditioning. The argument that sweet-savory combinations are inherently bad doesn't hold up - we enjoy honey-glazed ham and cranberry sauce with turkey. I still prefer pizza without it, but I can see the culinary logic behind it. The stronger argument against it might be textural - the moisture can make the crust soggy."

---

## Key Distinctions for Data

**Absolutist vs. Evaluativist**: Both take positions, but:
- Absolutist: "I'm right, you're wrong, end of discussion"
- Evaluativist: "Here's why I think this position is better supported, though I'm open to other evidence"

**Multiplist vs. Evaluativist**: Both acknowledge multiple perspectives, but:
- Multiplist: "All views are equally valid, there's no point comparing"
- Evaluativist: "Multiple views exist, and here's how I evaluate which is more justified"

## Labeling Instructions

For each sample, provide:
1. **stance**: One of [absolutist, multiplist, evaluativist]
2. **confidence**: One of [high, medium, low]
3. **justification**: Brief explanation (2-3 sentences) citing specific textual evidence
4. **key_markers**: List of specific phrases that indicate this stance

Consider:
- How does the writer treat their claims? (certain facts vs. personal opinions vs. evaluated judgments)
- How do they handle alternative perspectives? (dismiss vs. accept all vs. evaluate)
- Do they engage with counterarguments substantively?
- Do they show awareness of uncertainty or complexity?
- What would it take to change their mind?
"""

LABELING_TEMPLATE = """Please analyze the following post and classify its epistemic stance.

## Context
- Word Count: {word_count}

## Text to Analyze
---
{text}
---

## Your Analysis

Provide your analysis in the following JSON format:
```json
{{
    "stance": "[absolutist|multiplist|evaluativist]",
    "confidence": "[high|medium|low]",
    "justification": "[Your 2-3 sentence explanation citing specific evidence from the text]",
    "key_markers": ["list", "of", "specific", "phrases", "from", "the", "text"],
    "reasoning_quality": "[high|medium|low]",
    "engagement_with_alternatives": "[none|dismissive|superficial|substantive]"
}}
```

Additional fields explanation:
- **reasoning_quality**: How well-developed is the reasoning? (high = clear logical structure, evidence cited; low = assertions without support)
- **engagement_with_alternatives**: How does the writer handle other viewpoints?
  - none: No mention of alternatives
  - dismissive: Mentions but dismisses without engagement
  - superficial: Acknowledges but doesn't really engage
  - substantive: Genuinely engages with and evaluates alternatives

Remember:
- Base classification on HOW they argue, not WHAT they argue
- A person can have a strong opinion and still be evaluativist (if they reason well and acknowledge uncertainty)
- A person can seem open-minded but still be multiplist (if they refuse to evaluate competing claims)
"""


# ============================================================================
# LABELING FUNCTIONS
# ============================================================================

def create_labeling_message(sample_data):
    """Create the labeling prompt for a single sample."""
    
    return LABELING_TEMPLATE.format(
        word_count=sample_data.get('word_count', len(sample_data['text'].split())),
        text=sample_data['text']
    )


def label_single_sample(client, sample_data, model="claude-sonnet-4-20250514"):
    """
    Label a single sample using Claude API.
    
    Args:
        client: Anthropic client instance
        sample_data: Dictionary with sample information
        model: Claude model to use
    
    Returns:
        Dictionary with sample_id and labeling results
    """
    message_content = create_labeling_message(sample_data)
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": message_content}
            ]
        )
        
        # Extract JSON from response
        response_text = response.content[0].text
        
        # Try to parse JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = {
                "stance": "error",
                "confidence": "low",
                "justification": f"Could not parse response: {response_text[:200]}",
                "key_markers": [],
                "reasoning_quality": "unknown",
                "engagement_with_alternatives": "unknown"
            }
        
        return {
            "sample_id": sample_data.get('sample_id', 'unknown'),
            **result
        }
        
    except Exception as e:
        return {
            "sample_id": sample_data.get('sample_id', 'unknown'),
            "stance": "error",
            "confidence": "low",
            "justification": f"API error: {str(e)}",
            "key_markers": [],
            "reasoning_quality": "unknown",
            "engagement_with_alternatives": "unknown"
        }


def label_batch(client, samples, model="claude-sonnet-4-20250514", 
                delay_between_calls=0.5, progress_callback=None):
    """
    Label a batch of samples.
    
    Args:
        client: Anthropic client instance
        samples: List of sample dictionaries
        model: Claude model to use
        delay_between_calls: Seconds to wait between API calls
        progress_callback: Optional function to call with progress updates
    
    Returns:
        List of labeling results
    """
    results = []
    
    for i, sample in enumerate(samples):
        result = label_single_sample(client, sample, model)
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, len(samples), result)
        
        # Rate limiting
        if i < len(samples) - 1:
            time.sleep(delay_between_calls)
    
    return results


def save_results(results, samples, output_path):
    """
    Save labeling results merged with original sample data.
    """
    # Create lookup for results
    results_lookup = {r['sample_id']: r for r in results}
    
    # Merge with original data
    merged = []
    for sample in samples:
        sample_id = sample.get('sample_id', 'unknown')
        result = results_lookup.get(sample_id, {})
        
        merged.append({
            **sample,
            'epistemic_stance': result.get('stance', ''),
            'stance_confidence': result.get('confidence', ''),
            'stance_justification': result.get('justification', ''),
            'stance_markers': json.dumps(result.get('key_markers', [])),
            'reasoning_quality': result.get('reasoning_quality', ''),
            'engagement_with_alternatives': result.get('engagement_with_alternatives', ''),
        })
    
    # Save to CSV
    df_columns = ['sample_id', 'word_count', 'epistemic_stance', 'stance_confidence',
                  'reasoning_quality', 'engagement_with_alternatives',
                  'stance_justification', 'stance_markers', 'text']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=df_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged)
    
    print(f"Saved {len(merged)} labeled samples to {output_path}")
    
    # Print distribution
    print("\n" + "="*60)
    print("LABELING RESULTS SUMMARY")
    print("="*60)
    
    stance_counts = {}
    for m in merged:
        stance = m.get('epistemic_stance', 'unknown')
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
    
    print("\nStance distribution:")
    for stance, count in sorted(stance_counts.items()):
        print(f"  {stance}: {count} ({100*count/len(merged):.1f}%)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main labeling workflow."""
    import pandas as pd
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Label epistemic stance for text samples using Claude API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python label_stance.py relationship_advice_labeling_sample.csv
  python label_stance.py cmv_labeling_sample.csv --output custom_output.csv
  python label_stance.py data.csv --pilot-size 100
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to CSV file containing samples to label'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: derived from input filename)'
    )
    parser.add_argument(
        '--pilot-size',
        type=int,
        default=25,
        help='Number of samples to label in pilot batch (default: 25)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-20250514',
        help='Claude model to use (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        # Remove common suffixes like "_labeling_sample" or "_sample" and add "_labeled"
        stem = input_path.stem
        # Try to clean up common patterns
        for suffix in ['_labeling_sample', '_sample', '_labeling']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        output_path = input_path.parent / f"{stem}_labeled.csv"
    else:
        output_path = Path(args.output)
    
    # Initialize client
    client = Anthropic()
    
    # Load sample
    print(f"Loading labeling sample from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    samples = df.to_dict('records')
    
    print(f"Loaded {len(samples)} samples for labeling")
    
    # Start with pilot batch
    pilot_size = min(args.pilot_size, len(samples))
    pilot_samples = samples[:pilot_size]
    
    print(f"\nLabeling pilot batch of {pilot_size} samples...")
    
    def progress(current, total, result):
        stance = result.get('stance', 'unknown')
        conf = result.get('confidence', 'unknown')
        print(f"  [{current}/{total}] {result['sample_id'][:20]}...: {stance} ({conf})")
    
    results = label_batch(
        client, 
        pilot_samples, 
        model=args.model,
        delay_between_calls=args.delay,
        progress_callback=progress
    )
    
    # Save pilot results
    save_results(results, pilot_samples, str(output_path))
    
    print("\n" + "="*60)
    print("PILOT COMPLETE")
    print("="*60)
    print(f"""
    Review {output_path} to validate the labeling approach.
    
    Check:
    1. Are the stance classifications reasonable?
    2. Do the justifications cite appropriate evidence?
    3. Is the multiplist category being captured?
    
    Key questions:
    - What linguistic patterns distinguish the stances?
    
    Once validated, scale to full batch by modifying --pilot-size or
    running again with a larger value.
    """)


if __name__ == "__main__":
    main()
