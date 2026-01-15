import pandas as pd

df = pd.read_csv('wildchat_silver_labels.csv')

# Keep high-confidence evaluativist/absolutist
high_conf = df[
    (df['silver_label'] != 'multiplist') & 
    (df['silver_confidence'] >= 0.80)
]

# Lower threshold for multiplist
multiplist = df[
    (df['silver_label'] == 'multiplist') & 
    (df['silver_confidence'] >= 0.60)
]

combined = pd.concat([high_conf, multiplist])
combined.to_csv('silver_labels_wildchat_balanced.csv', index=False)

print("Label distribution:")
print(combined['silver_label'].value_counts())
print(f"\nTotal samples: {len(combined)}")