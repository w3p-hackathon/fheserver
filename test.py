import pandas as pd
import numpy as np
from concrete.ml.sklearn import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

# === Step 1: Load and encode allele pairs numerically ===
def load_encoded_snp(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['allele1'].isin(base_map) & df['allele2'].isin(base_map)]

    def encode(row):
        a1 = base_map[row['allele1']]
        a2 = base_map[row['allele2']]
        return int(f"{a1}{a2}")  # e.g., A=1, G=3 → "13" → 13

    df['genotype_code'] = df.apply(encode, axis=1)
    return df


def train_and_predict(X, y):
    model = LogisticRegression(n_bits=3)
    model.fit(X, y)
    model.compile(X)
    pred = model.predict(X)
    return pred

# === Main ===
if __name__ == "__main__":
    FILE = "snp_data.txt"     # Replace with actual file
    TARGET_RSID = "rs3131972"          # SNP you want to predict

    print("📥 Reading SNP data...")
    df = load_encoded_snp(FILE)
    df = df.set_index('rsid')
    x1 = df['genotype_code'].copy()
    X = pd.DataFrame(x1)
    X = x1.values.reshape(-1, 1).astype(np.float32) 
    y = df["chromosome"]
    print("🔐 Training encrypted model...")
    y_pred = train_and_predict(X, y)

    print(f"✅ Encrypted prediction: {len(y.tolist())}")
    # print("🎯 True values:", y.tolist())

