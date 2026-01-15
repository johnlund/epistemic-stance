import pandas as pd

gold = pd.read_csv('socialskills_multiplist_labeled-2806.csv')
silver = pd.read_csv('all_silver_combined.csv')

# Standardize gold
gold['label'] = gold['epistemic_stance']
gold['data_source'] = 'gold'
gold['sample_weight'] = 2.0

# Combine
combined = pd.concat([
    gold[['sample_id', 'text', 'label', 'data_source', 'sample_weight']],
    silver[['sample_id', 'text', 'label', 'data_source', 'sample_weight']]
], ignore_index=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined.to_csv('final_training_data.csv', index=False)

print(combined['data_source'].value_counts())
print(combined['label'].value_counts())