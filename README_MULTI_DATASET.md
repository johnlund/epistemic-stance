# Epistemic Stance Classification Pipeline

A complete pipeline for training a classifier to identify epistemic stances (Absolutist, Multiplist, Evaluativist) using a multi-dataset approach to ensure balanced training data.

## Overview

This project addresses a key challenge in epistemic stance classification: **different genres produce different stance distributions**. Our solution uses multiple datasets, each selected for its strength in representing a particular stance.

| Stance | Primary Dataset | Why It Works |
|--------|----------------|--------------|
| **Absolutist** | PERSUADE Essays | Students taught to argue definitively |
| **Multiplist** | r/relationship_advice | "Only you can decide" - defers judgment |
| **Evaluativist** | CMV Delta Responses | Must reason well to change minds |

## Theoretical Framework

Based on Kuhn, Cheney, and Weinstock (2000) "The Development of Epistemological Understanding":

| Stance | View of Knowledge | How Claims are Treated | Handling of Alternatives |
|--------|------------------|----------------------|-------------------------|
| **Absolutist** | Certain, fixed | Facts - correct or incorrect | Dismissed as wrong |
| **Multiplist** | Subjective, personal | Opinions - all equally valid | Accepted without evaluation |
| **Evaluativist** | Uncertain but evaluable | Judgments - compared via evidence | Evaluated for relative merit |

## The Multi-Dataset Challenge

### Why Single Datasets Fail

| Dataset | Observed Distribution | Problem |
|---------|----------------------|---------|
| PERSUADE Essays | ~60% Absolutist, <2% Multiplist | Genre rewards definitive arguments |
| ChangeMyView | ~80% Evaluativist | Persuasion requires reasoning |
| WildChat (LLM) | ~90% Absolutist | Transactional, confident responses |

### Our Solution

```
Training Data Composition (Target: 3,000 samples)
├── Absolutist (~1,000 samples)
│   └── PERSUADE argumentative essays
│       - Students writing persuasive essays
│       - Strong claims, definitive conclusions
│       - You have existing labels from prior work
│
├── Multiplist (~1,000 samples)
│   └── r/relationship_advice Reddit comments
│       - "Only you can decide" responses
│       - Defers judgment, treats all options as valid
│       - Filtered using multiplist linguistic patterns
│
└── Evaluativist (~1,000 samples)
    └── ChangeMyView delta-awarded responses
        - Successfully changed someone's mind
        - Engaged with reasoning substantively
        - Weighed evidence, acknowledged complexity
```

## Project Structure

```
epistemic-stance-classifier/
├── Data Exploration
│   ├── explore_cmv.py                    # CMV dataset (evaluativist)
│   └── explore_relationship_advice.py   # Reddit advice (multiplist)
│
├── Labeling
│   ├── label_cmv_stance.py              # Labels CMV data
│   └── label_multiplist_stance.py       # Labels advice data (multiplist-focused)
│
├── Training
│   └── train_classifier.py              # DeBERTa classifier training
│
└── Documentation
    └── README.md                         # This file
```

## Pipeline Steps

### Step 1: Prepare Evaluativist Data (CMV)

```bash
python explore_cmv.py
```

This extracts ChangeMyView posts, prioritizing:
- Delta-awarded responses (successful persuasion)
- Original posts with reasoning
- Filtering for substantive content (50+ words, reasoning indicators)

Output: `cmv_labeling_sample.csv`

### Step 2: Prepare Multiplist Data (relationship_advice)

```bash
python explore_relationship_advice.py
```

This extracts Reddit comments with multiplist indicators:
- "Only you can decide"
- "Everyone's situation is different"  
- "There's no right or wrong answer"
- Avoids comments with evaluative content

Output: `relationship_advice_labeling_sample.csv`

### Step 3: Label Each Dataset

```bash
# Label CMV data (evaluativist focus)
python label_cmv_stance.py

# Label relationship_advice data (multiplist focus)
python label_multiplist_stance.py
```

**Important**: The multiplist labeler distinguishes between:
- **True multiplist**: Refuses to evaluate, treats all options as equally valid
- **Deceptive multiplist**: Provides evaluation, then adds deferential ending ("but that's just me")

The latter is actually evaluativist, not multiplist.

### Step 4: Combine with PERSUADE Data

You have existing PERSUADE labels (~400 essays, ~60% absolutist).

```python
import pandas as pd

# Load each dataset
cmv_df = pd.read_csv("cmv_labeled.csv")
advice_df = pd.read_csv("relationship_advice_labeled.csv")
persuade_df = pd.read_csv("persuade_labeled.csv")  # Your existing data

# Filter to target stance (with some overlap for validation)
absolutist = persuade_df[persuade_df['stance'] == 'absolutist']
multiplist = advice_df[advice_df['epistemic_stance'] == 'multiplist']
evaluativist = cmv_df[cmv_df['epistemic_stance'] == 'evaluativist']

# Combine
combined = pd.concat([absolutist, multiplist, evaluativist])
combined.to_csv("combined_training_data.csv", index=False)
```

### Step 5: Train Classifier

```bash
python train_classifier.py \
    --input combined_training_data.csv \
    --output ./epistemic_classifier \
    --epochs 5 \
    --batch_size 16
```

## Key Insights

### The Multiplist Detection Challenge

Multiplist is the hardest stance to find because:

1. **It's socially passive** - People who refuse to evaluate don't drive engagement
2. **It's often disguised** - "I think X, but only you can decide" is evaluativist, not multiplist
3. **It's conversationally rare** - Online discourse rewards strong opinions

Our solution uses:
- Aggressive pattern matching for multiplist language
- A specialized labeling prompt that catches "deceptive multiplist"
- Relationship advice as a source (advice-giving naturally produces deference)

### Deceptive Multiplist

Many responses LOOK multiplist but are actually evaluativist:

❌ **Deceptive Multiplist** (actually evaluativist):
> "Honestly, I think you should talk to him about it - communication is key. But at the end of the day, only you know your relationship."

✅ **True Multiplist**:
> "This is such a personal decision. Some people would stay, some would leave - there's no right answer. Only you know what you can live with."

The difference: Does deference come AFTER evaluation (evaluativist) or INSTEAD OF evaluation (multiplist)?

## Cost Estimates

| Stage | Samples | Estimated Cost |
|-------|---------|----------------|
| CMV pilot | 50 | ~$1 |
| CMV full | 1,000 | ~$12 |
| Relationship_advice pilot | 50 | ~$1 |
| Relationship_advice full | 1,000 | ~$12 |
| PERSUADE (existing) | 400 | Already done |
| **Total labeling** | | **~$26** |
| Classifier training | - | ~$20-50 (Lambda) |

## Validation Strategy

After labeling, verify:

1. **Distribution balance**: ~33% each stance in combined data
2. **Cross-genre validation**: Some CMV examples should be absolutist/multiplist (not just evaluativist)
3. **True multiplist rate**: What % of "multiplist candidates" are actually multiplist?

## Requirements

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

## Research Applications

Once trained, this classifier enables:

| Research Question | How to Answer |
|------------------|---------------|
| What's the epistemic stance of LLMs? | Apply to WildChat/LMSYS responses |
| Do LLMs adapt to user stance? | Compare AI responses across user stance types |
| Does stance protect against manipulation? | Correlate stance with persuasion outcomes |
| How does stance vary by domain? | Apply across topics/contexts |

## References

- Kuhn, D., Cheney, R., & Weinstock, M. (2000). The development of epistemological understanding. *Cognitive Development*, 15(3), 309-328.

- Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. *WWW 2016*.

## License

MIT License
