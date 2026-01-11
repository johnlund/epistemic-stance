# Epistemic Stance Classification Pipeline (CMV Version)

A complete pipeline for training a model to identify epistemic stances (Absolutist, Multiplist, Evaluativist) in argumentative/dialogic text, based on Kuhn et al. (2000) framework.

## Overview

This project creates a classifier that can identify the epistemic stance of speakers in argumentative conversations. The pipeline:

1. **Downloads and filters** the ChangeMyView (CMV) dataset for high-quality reasoning samples
2. **Labels a subset** using Claude API with a carefully designed prompt
3. **Trains a classifier** (DeBERTa) on the labeled data
4. **Applies the classifier** to analyze epistemic patterns in human-AI conversations

## Why ChangeMyView?

CMV is ideal for epistemic stance classification because:

| Feature | Why It Matters |
|---------|---------------|
| **Explicit argumentation** | Users state positions with reasoning, not just requests |
| **Engagement required** | Responders must engage with arguments to earn deltas |
| **Persuasion outcomes** | Delta awards show when arguments successfully changed minds |
| **Natural language** | Real conversations, not formal essays |
| **Stance diversity** | Contains absolutist, multiplist, AND evaluativist examples |

### Expected Stance Distribution

Unlike argumentative essays (which skew 60%+ absolutist), CMV should show more balance:

- **Absolutist**: OPs with rigid positions, dismissive responses
- **Multiplist**: "Everyone's entitled to their opinion" responses
- **Evaluativist**: Delta-awarded responses (they convinced someone through reasoning!)

## Theoretical Framework

Based on Kuhn, Cheney, and Weinstock (2000) "The Development of Epistemological Understanding":

| Stance | View of Knowledge | How Claims are Treated | Handling of Alternatives |
|--------|------------------|----------------------|-------------------------|
| **Absolutist** | Certain, fixed | Facts - correct or incorrect | Dismissed as wrong |
| **Multiplist** | Subjective, personal | Opinions - all equally valid | Accepted without evaluation |
| **Evaluativist** | Uncertain but evaluable | Judgments - compared via evidence | Evaluated for relative merit |

## Project Structure

```
epistemic-stance-classifier/
├── explore_cmv.py           # CMV dataset exploration and filtering
├── label_cmv_stance.py      # Claude API labeling for CMV
├── train_classifier.py      # Classifier training (unchanged)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

### Requirements

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"
```

### requirements.txt

```
anthropic>=0.18.0
convokit>=3.0.0
datasets>=2.16.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.36.0
scikit-learn>=1.3.0
accelerate>=0.25.0
tqdm>=4.66.0
```

## Pipeline Steps

### Step 1: Explore and Filter CMV Dataset

```bash
python explore_cmv.py
```

This script:
- Downloads CMV via ConvoKit (includes delta labels)
- Analyzes dataset structure and statistics
- Filters for high-quality reasoning samples (50-1500 words, reasoning indicators)
- Creates stratified sample prioritizing:
  - Delta-awarded responses (likely evaluativist)
  - Original posts (stated positions)
  - Regular responses (varied stances)
- Saves to `cmv_labeling_sample.csv`

**Expected output:**
- ~3,000 high-quality reasoning samples
- Balanced across sample types
- Over-sampled delta-awarded responses

### Step 2: Label with Claude API

```bash
python label_cmv_stance.py
```

This script:
- Loads the labeling sample
- Labels each sample using Claude with the epistemic stance prompt
- Outputs additional fields:
  - `reasoning_quality`: How well-developed is the argumentation?
  - `engagement_with_alternatives`: How do they handle other viewpoints?
- Saves results to `cmv_pilot_labeled.csv` (pilot) or `cmv_labeled.csv` (full)

**Labeling prompt features:**
- Adapted for dialogic/argumentative text
- Clear examples from CMV context
- Distinguishes delta-awarded vs regular responses
- Captures reasoning quality and engagement patterns

**Cost estimate:**
- Pilot (50 samples): ~$1
- Full batch (3,000 samples): ~$30-40

### Step 3: Validate Pilot Labels

Before scaling, manually review the pilot labels:

```python
import pandas as pd

df = pd.read_csv("cmv_pilot_labeled.csv")

# Check distribution
print(df['epistemic_stance'].value_counts())

# Compare delta vs non-delta
delta_df = df[df['is_delta_awarded'] == True]
print("\nDelta-awarded responses:")
print(delta_df['epistemic_stance'].value_counts(normalize=True))

# Review examples
for stance in ['absolutist', 'multiplist', 'evaluativist']:
    print(f"\n=== {stance.upper()} EXAMPLES ===")
    examples = df[df['epistemic_stance'] == stance].sample(min(3, len(df[df['epistemic_stance'] == stance])))
    for _, row in examples.iterrows():
        print(f"\n[{row['sample_type']}] {row['text'][:300]}...")
        print(f"Justification: {row['stance_justification']}")
```

**Key validation questions:**
1. Are delta-awarded responses more likely to be evaluativist?
2. Are OPs with rigid views classified as absolutist?
3. Is the multiplist category being captured?
4. Do justifications cite appropriate textual evidence?

### Step 4: Train Classifier

After validating and completing full labeling:

```bash
python train_classifier.py \
    --input cmv_labeled.csv \
    --output ./epistemic_classifier \
    --epochs 5 \
    --batch_size 16 \
    --min_confidence medium
```

**Training considerations:**
- CMV texts are longer than typical classification tasks
- Consider using Longformer or chunking for texts >512 tokens
- Class balance should be better than essay data

### Step 5: Apply to Human-AI Conversations

Once trained, apply the classifier to analyze epistemic patterns:

```python
from train_classifier import EpistemicStanceClassifier

classifier = EpistemicStanceClassifier("./epistemic_classifier/best_model")

# Analyze WildChat conversations
# Analyze LMSYS-Chat-1M (compare across models)
# Analyze OpenAssistant

# Research questions you can now answer:
# - What's the distribution of epistemic stances in AI responses?
# - Do different models (GPT-4 vs Claude vs Llama) differ?
# - Do users with different stances get different AI responses?
# - Does AI epistemic stance correlate with persuasion outcomes?
```

## Research Applications

### Your Research Questions

| Question | How This Pipeline Helps |
|----------|------------------------|
| Does epistemic stance protect against LLM manipulation? | Classify user stances, correlate with persuasion outcomes |
| Do LLMs adapt stance to the listener? | Compare AI responses to users with different stances |
| What is the inherent epistemic stance of LLMs? | Apply classifier to AI responses across contexts |
| Can stance be altered for resilience? | Pre/post intervention measurement |

### Hypothesis: Delta-Awarded Responses Are More Evaluativist

A key prediction from the epistemic stance literature: **evaluativists are more persuasive** because they engage substantively with arguments rather than dismissing them (absolutist) or refusing to evaluate (multiplist).

CMV's delta system lets us test this directly:
- If delta-awarded responses are more evaluativist, it validates the framework
- If not, we learn something interesting about online persuasion

## Cost Summary

| Stage | Samples | Estimated Cost |
|-------|---------|----------------|
| Pilot labeling | 50 | ~$1 |
| Full labeling | 3,000 | ~$35 |
| Classifier training | - | ~$20-50 (Lambda) |
| **Total** | | **~$60-85** |

## Troubleshooting

### ConvoKit download issues
```python
# Alternative: Use HuggingFace version
from datasets import load_dataset
ds = load_dataset("webis/conclugen", "topics")
```

### Long text handling
```python
# Option 1: Truncate (loses information)
max_length = 512

# Option 2: Use Longformer (handles 4096 tokens)
model_name = "allenai/longformer-base-4096"

# Option 3: Chunk and aggregate
# Split text, classify each chunk, majority vote
```

### Imbalanced classes
```python
# Use class weights in training
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0,1,2], y=train_labels)
```

## References

- Kuhn, D., Cheney, R., & Weinstock, M. (2000). The development of epistemological understanding. *Cognitive Development*, 15(3), 309-328.

- Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. *WWW 2016*.

## License

MIT License - feel free to use and adapt for your research.
