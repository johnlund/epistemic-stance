#!/usr/bin/env python3
"""
Convert the epistemic stance CSV to Axolotl-compatible JSONL format.

For Mistral-Small-24B-Instruct-2501 (Mistral Small 3):
- Uses Tekken V7 tokenizer
- Uses standard OpenAI-style messages format
- Axolotl will apply the tokenizer's built-in chat template

Usage:
    python prepare_data_for_axolotl.py \
        --input final_training_data_balanced.csv \
        --output final_training_data_formatted.jsonl
"""

import argparse
import json
import pandas as pd
from pathlib import Path

# System prompt for epistemic stance classification
SYSTEM_PROMPT = """You are an expert classifier trained to identify epistemic stances in text based on Kuhn's developmental epistemology framework.

## The Three Epistemic Stances

**ABSOLUTIST**
Core belief: Knowledge is CERTAIN. There is ONE RIGHT ANSWER.

How they argue:
- Present claims as objective truth, not as a perspective
- Dismiss opposing views as wrong or misinformed
- Use evidence as PROOF, not as support for judgment
- Show little genuine engagement with counterarguments

Justification pattern: By authority or "obviousness" – no reasoning deemed necessary.

Linguistic markers: "Obviously...", "The truth is...", "You need to...", "The only option is...", "Anyone who thinks X is wrong", "This proves..."

Example: "You need to leave him. This is a huge red flag and staying would be a mistake. Don't let him manipulate you."

---

**MULTIPLIST**
Core belief: Knowledge is SUBJECTIVE. All opinions are EQUALLY VALID.

How they argue:
- Frame claims as personal opinions without justification
- Avoid evaluating which perspective has more merit
- Treat disagreement as natural and unresolvable
- Present multiple views but refuse to weigh them

Justification pattern: By personal preference – "that's just how I feel."

Linguistic markers: "That's just my opinion", "Everyone's entitled to their view", "Who's to say what's right?", "It depends on the person", "Only you can know"

Example: "Some people think you should leave, others think you should stay. Only you know what's right for you."

---

**EVALUATIVIST**
Core belief: Knowledge is UNCERTAIN but some claims have MORE MERIT based on evidence and reasoning.

How they argue:
- Acknowledge their position is a judgment, not absolute truth
- Engage substantively with counterarguments
- Weigh competing perspectives and explain why one is more compelling
- Show calibrated confidence matched to complexity

Justification pattern: Multiple justifications, cross-checked – cites reasoning AND evidence, acknowledges limitations.

Linguistic markers: "The evidence suggests...", "On balance...", "I could be wrong, but...", "While I understand X, I find Y more compelling because...", "Based on what you've described..."

Example: "Based on what you've described, the pattern suggests he's not respecting your boundaries. I'd lean toward reconsidering the relationship, though you know details I don't."

---

## Key Distinctions

**Absolutist vs. Evaluativist**: Both take positions. Ask: Do they JUSTIFY with reasoning, or assert as fact?
- Absolutist: "You must leave him" (directive, no reasoning)
- Evaluativist: "Based on the pattern you described, leaving seems stronger because..." (reasoned)

**Multiplist vs. Evaluativist**: Both acknowledge perspectives. Ask: Do they WEIGH them?
- Multiplist: "Some say leave, some say stay. Only you know." (no weighing)
- Evaluativist: "Leaving protects your health; staying might work if he changes. I'd lean toward leaving." (weighs tradeoffs)

**Deference distinction**: 
- Multiplist defers INSTEAD OF evaluation: "Only you can decide."
- Evaluativist defers AFTER evaluation: "I'd lean toward X, but you know things I don't."

---

Respond with JSON: {"stance": "absolutist|multiplist|evaluativist", "confidence": "high|medium|low"}"""


def get_confidence(sample_weight: float) -> str:
    """Map sample weight to confidence level."""
    if sample_weight >= 2.0:
        return "high"  # Gold data
    elif sample_weight >= 1.0:
        return "medium"  # Silver data
    else:
        return "low"


def convert_to_chat_format(row: pd.Series) -> dict:
    """
    Convert a single row to chat format compatible with Axolotl's chat_template type.
    
    Uses 'from' and 'value' fields to match the Axolotl config's message_field_role
    and message_field_content settings.
    """
    
    confidence = get_confidence(row.get('sample_weight', 1.0))
    
    return {
        "conversations": [
            {
                "from": "system",
                "value": SYSTEM_PROMPT
            },
            {
                "from": "human",
                "value": f"Classify the epistemic stance in the following text:\n\n{row['text']}"
            },
            {
                "from": "gpt",
                "value": json.dumps({"stance": row['label'], "confidence": confidence})
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Axolotl JSONL format')
    parser.add_argument('--input', '-i', required=True, help='Input CSV path')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL path')
    
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Convert to JSONL
    print(f"Converting to chat format for Mistral-Small-24B-Instruct-2501...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            example = convert_to_chat_format(row)
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(df)} examples to {args.output}")
    
    # Print a sample
    print("\nSample output:")
    sample = convert_to_chat_format(df.iloc[0])
    print(json.dumps(sample, indent=2)[:500] + "...")
    
    # Verify format
    print("\nFormat verification:")
    print(f"  - Message roles: {[m['from'] for m in sample['conversations']]}")
    print(f"  - Response format: {sample['conversations'][-1]['value']}")


if __name__ == '__main__':
    main()
