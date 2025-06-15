import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from concrete.ml.sklearn import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import io

app = FastAPI(title="SNP Analysis API", description="API for SNP genotype analysis using encrypted ML")

base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

# === Pydantic Models for Request/Response ===
class SNPRecord(BaseModel):
    rsid: str
    allele1: str
    allele2: str
    chromosome: Optional[int] = None

class SNPDataRequest(BaseModel):
    snp_data: List[SNPRecord]

class PredictionResponse(BaseModel):
    success: bool
    message: str
    predictions: List[int]
    total_samples: int
    target_rsid: Optional[str] = None



def load_encoded_snp(file_data):
    df = pd.read_csv(io.StringIO(file_data), sep='\t')
    df = df[df['allele1'].isin(base_map) & df['allele2'].isin(base_map)]
    def encode(row):
        a1 = base_map[row['allele1']]
        a2 = base_map[row['allele2']]
        return int(f"{a1}{a2}")  # e.g., A=1, G=3 → "13" → 13

    df['genotype_code'] = df.apply(encode, axis=1)
    df = load_encoded_snp(file_data)
    df = df.set_index('rsid')
    
    return df

def train_and_predict(X, y):
    model = LogisticRegression(n_bits=3)
    model.fit(X, y)
    model.compile(X)
    pred = model.predict(X)
    return pred


@app.post("/analyze-snp", response_model=PredictionResponse)
async def analyze_snp_data(dataFile: UploadFile = File(...)):
    """z
    Analyze SNP data using encrypted machine learning
    
    - **snp_data**: List of SNP records with rsid, allele1, allele2, and optional chromosome
    - **target_rsid**: Optional specific SNP to focus analysis on
    """
    print("Processing SNP data...")
    # Encode the SNP data
    geneData = await dataFile.read()
    df = load_encoded_snp(geneData.decode("utf-8"))
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid SNP records after encoding")
    
    df = df.set_index('rsid')
    X = df['genotype_code'].values.reshape(-1, 1).astype(np.float32)
    
    y = df['chromosome'].fillna(1).astype(int)  # Fill NaN with 1 if needed
    
    print("Training encrypted model...")
    
    # Train and predict
    y_pred = train_and_predict(X, y)
    
    print(f"Analysis complete for {len(y)} samples")
    
    return PredictionResponse(
        success=True,
        message=f"Successfully analyzed {len(y)} SNP samples",
        predictions=[],
        total_samples=len(y),
        target_rsid=0#request.target_rsid
    )

# === Example usage in main ===
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting SNP Analysis API server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
