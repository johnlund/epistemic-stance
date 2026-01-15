# Classifier Error Analysis Report
==================================================

## Overall Performance
- Accuracy: 76.51%
- Total samples: 281
- Total errors: 66

## Per-Class Performance
### Evaluativist
- Accuracy: 86.84%
- Correct: 165/190
- Errors: 25

### Absolutist
- Accuracy: 59.21%
- Correct: 45/76
- Errors: 31

### Multiplist
- Accuracy: 33.33%
- Correct: 5/15
- Errors: 10

## Error Type Breakdown
- absolutist_as_evaluativist: 31
- evaluativist_as_absolutist: 21
- multiplist_as_evaluativist: 9
- evaluativist_as_multiplist: 4
- multiplist_as_absolutist: 1

## Confidence Analysis
- Mean confidence on errors: 0.774
- Mean confidence on correct: 0.858
- High-confidence errors (>0.8): 36
- Very high-confidence errors (>0.9): 16

## Linguistic Pattern Analysis

Patterns more common in errors vs correct predictions:
- hedging: errors=62.1%, correct=64.2% (↓)
- certainty: errors=42.4%, correct=53.0% (↓)
- relativism: errors=71.2%, correct=74.0% (↓)
- evidence: errors=13.6%, correct=5.6% (↑)
- counterargument: errors=83.3%, correct=81.4% (↑)

## Recommendations

⚠️ **Multiplist accuracy is low.** Consider:
  - Reviewing multiplist training samples for consistency
  - Increasing focal loss gamma
  - Data augmentation for multiplist class

⚠️ **High-confidence errors detected.** Consider:
  - Reviewing these samples manually
  - Checking for labeling inconsistencies
  - Using higher confidence threshold for silver labels
