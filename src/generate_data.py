import pandas as pd
import numpy as np

np.random.seed(42)
n = 100000
fraud_rate = 0.02

n_fraud = int(n * fraud_rate)
n_normal = n - n_fraud

def generate_transactions(n_samples, is_fraud):
    return pd.DataFrame({
        "amount": np.random.exponential(500 if is_fraud else 100, n_samples),
        "hour": np.random.choice(range(0, 6) if is_fraud else range(8, 22), n_samples),
        "distance_from_home": np.random.exponential(200 if is_fraud else 20, n_samples),
        "transactions_last_24h": np.random.randint(10, 20, n_samples) if is_fraud else np.random.randint(1, 4, n_samples),
        "different_location": np.random.choice([1, 0], n_samples, p=[0.9, 0.1] if is_fraud else [0.1, 0.9]),
        "is_fraud": int(is_fraud)
    })

fraud = generate_transactions(n_fraud, True)
normal = generate_transactions(n_normal, False)

df = pd.concat([fraud, normal]).sample(frac=1).reset_index(drop=True)
df.to_csv("data/transactions.csv", index=False)
print(f"✅ Dataset gerado: {len(df)} transações ({n_fraud} fraudes)")