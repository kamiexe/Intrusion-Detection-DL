import pandas as pd
import numpy as np

# Generate synthetic data with patterns
num_samples = 10_000

data = {
    'duration': np.random.randint(0, 1000, num_samples),
    'src_bytes': np.random.randint(0, 1e6, num_samples),
    'dst_bytes': np.random.randint(0, 1e6, num_samples),
    'count': np.random.randint(0, 50, num_samples),
    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], num_samples),
    'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh'], num_samples),
    'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], num_samples),
    'label': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Add realistic patterns: attacks have higher src_bytes
attack_mask = df['label'] == 1
df.loc[attack_mask, 'src_bytes'] = np.random.randint(1e5, 1e7, sum(attack_mask))

# Add attack types for multiclass (optional)
attack_types = ['normal', 'dos', 'probe', 'r2l']
df['attack_type'] = df['label'].apply(
    lambda x: 'normal' if x == 0 else np.random.choice(attack_types[1:])
)

df.to_csv('mock_ids_dataset.csv', index=False)