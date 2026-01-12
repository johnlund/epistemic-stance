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

SYSTEM_PROMPT = """You are an expert in epistemic cognition research, specifically trained to identify epistemic stances in argumentative and dialogic text based on the framework developed in the paper "The Development of Epistemological Understanding" by Kuhn, Cheney, and Weinstock (2000) and "The development of epistemological theories: Beliefs about knowledge and knowing and their relation to learning" by Hofer, B. K., & Pintrich, P. R. (1997). The ideas were expanded upon in the paper "Role of Epistemic Beliefs and Scientific Argumentation in Science Learning" by E. Michael Nussbaum, Gale M. Sinatra and Anne Poliquin (2008) and the paper "Strengthening Human Epistemic Agency in the Symbiotic Learning Partnership With Generative Artificial Intelligence" by Wu, Lee, Chai, and Tsai (2025).

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
- Engage in discussion "in order to find the right answer" (Nussbaum et al., 2008)

Linguistic markers:
- Certainty language: "It is a fact that...", "The truth is...", "Obviously...", "Clearly..."
- Dismissive phrases: "Anyone who thinks X is wrong", "There's no legitimate argument for..."
- Definitive claims: "This proves...", "This is undeniable...", "Without question..."
- Binary framing: Right/wrong, true/false, correct/incorrect
- Appeals to authority as final word: "Science has proven...", "Experts agree..." (without nuance)
- Directive language: "You need to...", "You must...", "The only option is..."

Justification pattern (Wu et al., 2025): Justification by authority or "obviousness"
- Cites experts/authority as the final word
- Treats claims as self-evident ("obviously," "clearly")
- No justification deemed necessary—it's just true

Example in context:
"Pineapple on pizza is objectively wrong. It's a fact that sweet and savory don't belong together. Anyone who enjoys it simply has bad taste. There's no real argument for it - it's just wrong."

Second example (showing "seeking the right answer" pattern):
"Look, I've done the research on this. Studies show that X causes Y. That's just the science. People who disagree are ignoring the evidence."

---

**MULTIPLIST**
Core belief: Knowledge is SUBJECTIVE. All opinions are EQUALLY VALID because everyone is entitled to their view.

How they argue:
- Frame claims as personal opinions without attempting justification
- Avoid evaluating which perspective has more merit
- Treat disagreement as natural and unresolvable
- Resist taking firm positions or making judgments
- May present multiple views but refuse to weigh them
- Show "tolerance for inconsistencies" (Nussbaum et al., 2008)
- In dialogic contexts: tend to agree, repeat, or disengage rather than probe

Linguistic markers:
- Opinion framing: "That's just my opinion", "Everyone's entitled to their view"
- Relativistic phrases: "It's all subjective", "Who's to say what's right?"
- Equivalence statements: "Both sides have valid points", "Neither is better or worse"
- Avoidance of judgment: "I'm not saying one is right", "It depends on the person"
- Deflection to individual: "That's for each person to decide", "There's no right answer"
- Pure deference without analysis: "Only you can know", "It's your call"

Justification pattern (Wu et al., 2025): Justification by personal preference
- "That's just how I feel"
- "Everyone has their own truth"
- Personal experience as sole warrant, not generalizable

Critical feature - Inconsistency tolerance:
Multiplists "did not seem to spot or be bothered by inconsistencies in their thinking... 
The fact that they accept—by virtue of being a multiplist—that opposing ideas can both 
be right might contribute to a tolerance for inconsistencies" (Nussbaum et al., 2008).
- May hold or present contradictory positions without acknowledging the tension
- Tolerates logical inconsistencies because "different perspectives"

Example in context:
"I mean, some people like pineapple on pizza and some don't. It's really just a matter of personal taste. Who am I to say what someone else should enjoy? Everyone's entitled to their own preferences. There's no objectively correct answer here."

Second example (showing inconsistency tolerance):
"Some people say you should definitely leave him, others say you should stay and work it out. Both perspectives are valid - it really depends on who you are as a person. Only you can decide what's right for you."

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
- In dialogic contexts: ask questions, request clarification, build on ideas (Nussbaum et al., 2008)
- Probe for more information before forming judgments

Linguistic markers:
- Qualified claims: "The evidence suggests...", "On balance...", "It seems likely that..."
- Engagement with alternatives: "While I understand the argument that X, I find Y more compelling because..."
- Acknowledgment of uncertainty: "I could be wrong, but...", "This is my current view..."
- Comparative evaluation: "This argument is stronger because...", "The evidence for X outweighs..."
- Metacognitive awareness: "I've reconsidered...", "You've made me think about..."
- Conditional reasoning: "If we accept X, then...", "Given Y, it follows that..."
- Calibrated confidence: More certain about clear-cut issues, less about complex ones

Justification pattern (Wu et al., 2025): Multiple justifications, cross-checked
- Cites reasoning AND evidence
- Acknowledges limitations of sources
- Integrates multiple types of support

Example in context:
"I've always disliked pineapple on pizza, but I recognize that's partly cultural conditioning. The argument that sweet-savory combinations are inherently bad doesn't hold up - we enjoy honey-glazed ham and cranberry sauce with turkey. I still prefer pizza without it, but I can see the culinary logic behind it. The stronger argument against it might be textural - the moisture can make the crust soggy."

Second example (showing weighing and calibrated deference):
"Based on what you've described, leaving seems like the stronger option—the pattern of broken promises suggests he's not committed to change. That said, couples therapy has helped some people in similar situations. I'd lean toward leaving, but you know details about his recent behavior that I don't."

---

## Key Distinctions for Coding

### Absolutist vs. Evaluativist

Both take positions, but differ in how they justify them.

According to Nussbaum et al. (2008), absolutists "believe in only one right answer" and 
engage in discussion "in order to find the right answer," but "may probe less deeply than 
evaluativists" and struggle to "address and eliminate misconceptions." Wu et al. (2025) 
note that absolutists accept authority without questioning—"no justification deemed necessary."

Key question: Does the writer JUSTIFY their position with reasoning, or assert it as fact?

ABSOLUTIST indicators:
- States conclusions as certain facts without supporting reasoning
- Appeals to authority ("experts say," "it's common knowledge," "obviously")
- Treats disagreement as simply being wrong rather than having different evidence
- Uses definitive language: "you must," "you need to," "the only option," "clearly"
- Dismisses alternative views rather than engaging with them
- Shows certainty disproportionate to the complexity of the issue

EVALUATIVIST indicators:
- Provides reasoning for why they hold their position
- Acknowledges the position could be revised with new evidence
- Engages with counterarguments rather than dismissing them
- Uses hedged language: "based on what you've said," "it seems like," "I'd argue"
- Distinguishes between stronger and weaker evidence
- Shows calibrated confidence (more certain about clear-cut issues, less about complex ones)

Example distinction:
- ABSOLUTIST: "You need to leave him. This is a huge red flag and staying would be 
  a mistake. Don't let him manipulate you."
- EVALUATIVIST: "Based on what you've described, the pattern of behavior suggests 
  he's not respecting your boundaries. That's concerning because healthy relationships 
  require mutual respect. I'd lean toward reconsidering the relationship, though you 
  know details I don't."

---

### Multiplist vs. Evaluativist

This is often the hardest distinction. Both acknowledge multiple perspectives exist.

According to Nussbaum et al. (2008), multiplists show "tolerance for inconsistencies" 
and treat opposing views as equally valid. Evaluativists critically weigh claims 
against each other. Multiplists "interacted with their partners less" and "tended to 
just repeat or agree," while evaluativists "raised more issues" and were "more critical."

Key question: Does the writer WEIGH the options against each other?

MULTIPLIST indicators:
- Presents options without comparing their relative merit
- Tolerates or ignores contradictions between viewpoints
- Treats all perspectives as equally valid ("everyone's right in their own way")
- Defers judgment because "it's personal" rather than because evidence is mixed
- Doesn't ask clarifying questions or challenge claims
- May hold contradictory positions without acknowledging the tension

EVALUATIVIST indicators:
- Compares options using criteria (even informal ones)
- Notes tensions or trade-offs between viewpoints
- Suggests one option might be "better" or "stronger" for certain reasons
- Defers judgment because the evidence is genuinely uncertain/mixed (after weighing)
- Asks questions, requests clarification, probes for more information
- Acknowledges and tries to resolve contradictions

Example distinction:
- MULTIPLIST: "Some people think you should leave, others think you should stay. 
  Only you know what's right for you."
- EVALUATIVIST: "Leaving might protect your mental health, but staying could 
  preserve the relationship if he's willing to change. Given what you've described, 
  I'd lean toward leaving, but ultimately you know more about the situation than I do."

---

### The Deference Distinction

Both multiplists and evaluativists may defer to the reader/listener, but for different reasons:

MULTIPLIST deference: "I can't judge, it's personal"
- Defers because they believe NO judgment is possible
- Refuses to weigh options at all
- Deference comes INSTEAD OF evaluation
- Example: "This is really personal. Only you can decide."

EVALUATIVIST deference: "I've weighed this, but you have information I don't"
- Defers because they've reached the LIMITS of their analysis
- Has already provided evaluation before deferring
- Deference comes AFTER evaluation
- Example: "Based on what you've shared, X seems stronger, but you know things I don't."

---

## Note on Domain Sensitivity

Kuhn et al. (2000) found that epistemic stances can vary by domain (personal taste vs. 
values vs. empirical truth). When coding, consider whether the stance is APPROPRIATE 
to the domain being discussed:

- Personal taste (favorite color, food preferences): Multiplist stance may be appropriate
- Values and ethics (what's morally right): Mixed—some evaluation possible
- Empirical claims (does this treatment work?): Evaluativist stance most appropriate

Watch for:
- Strong certainty about genuinely uncertain/complex issues → likely ABSOLUTIST
- Strong relativism about evidence-based issues → likely MULTIPLIST  
- Calibrated confidence matched to domain complexity → likely EVALUATIVIST

---

## Labeling Instructions

For each sample, provide:
1. **stance**: One of [absolutist, multiplist, evaluativist]
2. **confidence**: One of [high, medium, low]
3. **justification**: Brief explanation (2-3 sentences) citing specific textual evidence
4. **key_markers**: List of specific phrases that indicate this stance

Consider:
- How does the writer treat their claims? (certain facts vs. personal opinions vs. evaluated judgments)
- How do they JUSTIFY their claims? (authority/obviousness vs. personal preference vs. reasoned evidence)
- How do they handle alternative perspectives? (dismiss vs. accept all equally vs. weigh and compare)
- Do they engage with counterarguments substantively?
- Do they show awareness of uncertainty or complexity?
- Do they tolerate or try to resolve inconsistencies?
- Is their confidence level calibrated to the domain and complexity of the issue?
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
        default=50,
        help='Number of samples to label in pilot batch (default: 50)'
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
