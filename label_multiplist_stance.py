"""
Epistemic Stance Labeling for Multiplist Candidates
====================================================

This module contains the labeling prompt and API code specifically tuned
for identifying multiplist reasoning in relationship advice and similar contexts.

The key challenge: distinguishing TRUE multiplist from "soft evaluativist"

TRUE MULTIPLIST:
- "Everyone's situation is different, only you can decide"
- "There's no right or wrong here, it's personal"
- Refuses to weigh or compare options

SOFT EVALUATIVIST (not multiplist):
- "I think X is better, but ultimately only you can decide"
- "Here's my reasoning... but everyone's different"
- Provides evaluation, then defers

Based on Kuhn et al. (2000) framework.
"""

import json
import csv
import time
from anthropic import Anthropic

# ============================================================================
# LABELING PROMPT - TUNED FOR MULTIPLIST DETECTION
# ============================================================================

SYSTEM_PROMPT = """You are an expert in epistemic cognition research, trained to identify epistemic stances based on Kuhn, Cheney, and Weinstock (2000). Your special focus is distinguishing GENUINE MULTIPLIST reasoning from other stances.

## The Three Epistemic Stances

### ABSOLUTIST
Core belief: Knowledge is CERTAIN. There is ONE RIGHT ANSWER.

In advice contexts, absolutists:
- Give definitive directives: "You need to leave him" / "You should definitely X"
- Dismiss the other perspective as clearly wrong
- Present their advice as the obvious/only correct choice
- Show no uncertainty or acknowledgment of complexity

Markers: "obviously", "clearly", "you need to", "you must", "there's no question", "red flag", "deal breaker"

### MULTIPLIST  
Core belief: Knowledge is SUBJECTIVE. All perspectives are EQUALLY VALID.

In advice contexts, multiplists:
- REFUSE to evaluate which option is better
- Treat all perspectives as equally valid WITHOUT weighing them
- Deflect judgment back to the asker: "only you can decide"
- Avoid providing reasoning for preferring one option
- May describe options but WON'T compare their merit

**CRITICAL DISTINCTION**: True multiplist is NOT the same as:
- Giving an opinion then adding "but that's just me" (that's evaluativist + humility)
- Explaining pros/cons then saying "only you can decide" (that's evaluativist + deference)
- Acknowledging complexity while still expressing a preference (that's evaluativist)

TRUE multiplist markers:
- "There's no right or wrong answer here"
- "Only you know what's best for you"
- "Everyone's situation is different" (without then evaluating THIS situation)
- "That's a personal choice only you can make"
- "Who am I to say what you should do"

### EVALUATIVIST
Core belief: Knowledge is UNCERTAIN but some positions have MORE MERIT based on evidence/reasoning.

In advice contexts, evaluativists:
- Weigh options and explain why one might be better
- Acknowledge uncertainty while still providing assessment
- Consider multiple perspectives but evaluate their relative merit
- May express a preference with reasoning, even if tentative
- Often end with deference ("but ultimately your call") AFTER providing evaluation

Markers: "on balance", "I think X because...", "while I understand Y, X seems...", "considering...", "the stronger argument is..."

## The Hard Cases: Multiplist vs. Evaluativist

Many responses LOOK multiplist but are actually evaluativist. Here's how to tell:

**EVALUATIVIST disguised as multiplist:**
"I've been in a similar situation. Honestly, I think you should talk to him about it - communication is key in relationships. But at the end of the day, only you know your relationship."
→ This is EVALUATIVIST. They provided an evaluation ("you should talk to him") with reasoning ("communication is key"), then added a deferential ending. The deference doesn't erase the evaluation.

**TRUE MULTIPLIST:**
"This is such a personal decision. Some people would stay, some would leave - there's no right answer. Only you know what you can live with and what matters most to you."
→ This is MULTIPLIST. No evaluation of which option is better. No reasoning for preferring one. Just acknowledgment that it's subjective.

**Another EVALUATIVIST disguised as multiplist:**
"It depends on what you value more. If stability is important to you, staying makes sense. If growth matters more, leaving might be better. Think about what you really want."
→ This is actually EVALUATIVIST. They're helping the person evaluate by providing a framework for comparison. They're not refusing to evaluate - they're scaffolding evaluation.

**TRUE MULTIPLIST:**
"Different things work for different people. There's no way for anyone to tell you what the right choice is - that's something only you can figure out for yourself."
→ MULTIPLIST. Refuses to provide any evaluative framework.

## Labeling Instructions

For each sample, provide:
1. **stance**: One of [absolutist, multiplist, evaluativist]
2. **confidence**: One of [high, medium, low]
3. **justification**: 2-3 sentences explaining your classification with specific textual evidence
4. **deceptive_multiplist**: Boolean - does this LOOK multiplist but is actually evaluativist?
5. **key_markers**: Specific phrases that indicate this stance

Pay special attention to:
- Does the response provide ANY evaluation or preference, even weak?
- Does it give reasoning for why one option might be better?
- Does deference ("only you can decide") come AFTER an evaluation, or INSTEAD of one?
- Does it provide a framework for the person to evaluate (which is itself evaluation)?
"""

LABELING_TEMPLATE = """Please analyze the following relationship advice comment and classify its epistemic stance.

## Context
- Source: r/relationship_advice Reddit comment
- Multiplist score (heuristic): {multiplist_score}
- Word count: {word_count}

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
    "deceptive_multiplist": [true|false],
    "provides_evaluation": [true|false],
    "provides_reasoning": [true|false],
    "defers_judgment": [true|false],
    "key_markers": ["list", "of", "specific", "phrases"]
}}
```

Field explanations:
- **deceptive_multiplist**: True if it uses multiplist language but actually provides evaluation
- **provides_evaluation**: True if it expresses any preference between options
- **provides_reasoning**: True if it gives reasons for preferring one option
- **defers_judgment**: True if it includes phrases like "only you can decide", "your choice"

Remember:
- Deference AFTER evaluation = evaluativist (not multiplist)
- Deference INSTEAD OF evaluation = multiplist
- Many samples will LOOK multiplist but actually be evaluativist - that's the key distinction to make"""


# ============================================================================
# LABELING FUNCTIONS
# ============================================================================

def create_labeling_message(sample_data):
    """Create the labeling prompt for a single sample."""
    return LABELING_TEMPLATE.format(
        multiplist_score=sample_data.get('multiplist_score', 'N/A'),
        word_count=sample_data.get('word_count', len(sample_data['text'].split())),
        text=sample_data['text']
    )


def label_single_sample(client, sample_data, model="claude-sonnet-4-20250514"):
    """
    Label a single sample using Claude API.
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
        
        response_text = response.content[0].text
        
        # Parse JSON from response
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
                "deceptive_multiplist": False,
                "provides_evaluation": None,
                "provides_reasoning": None,
                "defers_judgment": None,
                "key_markers": []
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
            "deceptive_multiplist": False,
            "provides_evaluation": None,
            "provides_reasoning": None,
            "defers_judgment": None,
            "key_markers": []
        }


def label_batch(client, samples, model="claude-sonnet-4-20250514", 
                delay_between_calls=0.5, progress_callback=None):
    """Label a batch of samples."""
    results = []
    
    for i, sample in enumerate(samples):
        result = label_single_sample(client, sample, model)
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, len(samples), result)
        
        if i < len(samples) - 1:
            time.sleep(delay_between_calls)
    
    return results


def save_results(results, samples, output_path):
    """Save labeling results merged with original sample data."""
    results_lookup = {r['sample_id']: r for r in results}
    
    merged = []
    for sample in samples:
        sample_id = sample.get('sample_id', 'unknown')
        result = results_lookup.get(sample_id, {})
        
        merged.append({
            **sample,
            'epistemic_stance': result.get('stance', ''),
            'stance_confidence': result.get('confidence', ''),
            'stance_justification': result.get('justification', ''),
            'deceptive_multiplist': result.get('deceptive_multiplist', ''),
            'provides_evaluation': result.get('provides_evaluation', ''),
            'provides_reasoning': result.get('provides_reasoning', ''),
            'defers_judgment': result.get('defers_judgment', ''),
            'stance_markers': json.dumps(result.get('key_markers', [])),
        })
    
    # Save to CSV
    df_columns = ['sample_id', 'dataset', 'sample_type', 'word_count',
                  'multiplist_score', 'epistemic_stance', 'stance_confidence',
                  'deceptive_multiplist', 'provides_evaluation', 'provides_reasoning',
                  'defers_judgment', 'stance_justification', 'stance_markers', 'text']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=df_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged)
    
    print(f"Saved {len(merged)} labeled samples to {output_path}")
    
    # Print summary
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
    
    # Deceptive multiplist analysis
    deceptive_count = sum(1 for m in merged if m.get('deceptive_multiplist') == True)
    print(f"\nDeceptive multiplist (looks multiplist, is evaluativist): {deceptive_count}")
    
    # True multiplist rate
    true_multiplist = sum(1 for m in merged 
                          if m.get('epistemic_stance') == 'multiplist' 
                          and not m.get('deceptive_multiplist'))
    print(f"True multiplist: {true_multiplist}")
    
    return merged


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main labeling workflow."""
    import pandas as pd
    
    client = Anthropic()
    
    print("Loading labeling sample...")
    df = pd.read_csv("relationship_advice_labeling_sample.csv")
    samples = df.to_dict('records')
    
    print(f"Loaded {len(samples)} samples for labeling")
    
    # Start with pilot
    pilot_size = 50
    pilot_samples = samples[:pilot_size]
    
    print(f"\nLabeling pilot batch of {pilot_size} samples...")
    print("(Looking for TRUE multiplist vs deceptive multiplist...)")
    
    def progress(current, total, result):
        stance = result.get('stance', 'unknown')
        deceptive = result.get('deceptive_multiplist', False)
        marker = "⚠️ DECEPTIVE" if deceptive else ""
        print(f"  [{current}/{total}] {result['sample_id']}: {stance} {marker}")
    
    results = label_batch(
        client,
        pilot_samples,
        model="claude-sonnet-4-20250514",
        delay_between_calls=0.5,
        progress_callback=progress
    )
    
    # Save results
    merged = save_results(results, pilot_samples, "relationship_advice_pilot_labeled.csv")
    
    print("\n" + "="*60)
    print("PILOT COMPLETE")
    print("="*60)
    print("""
    Review relationship_advice_pilot_labeled.csv to validate.
    
    Key questions:
    1. What % of "multiplist candidates" are ACTUALLY multiplist?
    2. How many are "deceptive multiplist" (evaluativist with deferential ending)?
    3. Do we have enough TRUE multiplist examples?
    
    If the yield is low, we may need to:
    - Adjust the multiplist patterns to be more specific
    - Accept that true multiplist is genuinely rare
    - Use class weighting in training
    
    Expected finding: Many "only you can decide" comments actually
    provide evaluation first, making them evaluativist, not multiplist.
    """)


if __name__ == "__main__":
    main()
