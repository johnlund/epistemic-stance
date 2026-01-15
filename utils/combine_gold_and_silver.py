import pandas as pd

# Load both datasets
gold = pd.read_csv('socialskills_multiplist_labeled-2806.csv')
silver = pd.read_csv('silver_labels_balanced.csv')

# Standardize gold columns
gold['label'] = gold['epistemic_stance']
gold['data_source'] = 'gold'
gold['sample_weight'] = 2.0  # Weight gold samples higher

# Standardize silver columns
silver['label'] = silver['silver_label']
silver['data_source'] = 'silver'
silver['sample_weight'] = silver['silver_confidence']  # Weight by confidence

# Select common columns
gold_subset = gold[['sample_id', 'text', 'label', 'data_source', 'sample_weight']]
silver_subset = silver[['sample_id', 'text', 'label', 'data_source', 'sample_weight']]

# Combine and shuffle
combined = pd.concat([gold_subset, silver_subset], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined.to_csv('combined_training_data.csv', index=False)

print("Combined dataset:")
print(combined['label'].value_counts())
print(f"\nTotal: {len(combined)}")
print(f"Gold: {(combined['data_source'] == 'gold').sum()}")
print(f"Silver: {(combined['data_source'] == 'silver').sum()}")