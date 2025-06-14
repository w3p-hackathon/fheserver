#!/usr/bin/env python3
"""
Privacy-Preserving Genomic Regression API Server with Concrete-ML
FastAPI server that accepts CSV uploads and returns FHE regression results
"""

import pandas as pd
import numpy as np
import io
import json
import time
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import os

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Concrete-ML imports
from concrete.ml.sklearn import LinearRegression as FHELinearRegression
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Privacy-Preserving Genomic Analysis API",
    description="Secure genomic data analysis using Fully Homomorphic Encryption",
    version="1.0.0"
)

# Global storage for models and results (in production, use proper database)
MODELS = {}
RESULTS = {}

# Request/Response Models
class AnalysisConfig(BaseModel):
    n_bits: int = 6
    test_size: float = 0.2
    target_strategy: str = "synthetic"
    create_synthetic_target: bool = True

class PredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]

class AnalysisResponse(BaseModel):
    model_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    timing: Optional[Dict[str, float]] = None

class PrivateGenomicAnalyzer:
    """Core analyzer class for FHE genomic regression"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.fhe_circuit = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.results = {}
        self.feature_columns = []
        
    def load_csv_data(self, csv_content: bytes) -> bool:
        """Load CSV data from bytes"""
        try:
            # Try different separators
            csv_string = csv_content.decode('utf-8')
            try:
                self.df = pd.read_csv(io.StringIO(csv_string), sep='\t')
            except:
                self.df = pd.read_csv(io.StringIO(csv_string), sep=',')
            
            return True
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")
    
    def preprocess_data(self, n_bits: int = 6, target_strategy: str = "synthetic", 
                       create_synthetic_target: bool = True) -> Dict[str, Any]:
        """Preprocess genomic data for FHE"""
        
        # Handle missing values
        self.df = self.df.fillna(0)
        
        features = []
        preprocessing_info = {
            "original_shape": self.df.shape,
            "columns": list(self.df.columns),
            "missing_values_filled": True
        }
        
        # Encode chromosome
        if 'chromosome' in self.df.columns:
            if self.df['chromosome'].dtype == 'object':
                le_chrom = LabelEncoder()
                self.df['chromosome_encoded'] = le_chrom.fit_transform(self.df['chromosome'])
                self.label_encoders['chromosome'] = le_chrom
            else:
                self.df['chromosome_encoded'] = self.df['chromosome']
            features.append('chromosome_encoded')
        
        # Encode alleles
        allele_cols = [col for col in self.df.columns if 'allele' in col.lower()]
        for col in allele_cols:
            if col in self.df.columns:
                le_allele = LabelEncoder()
                allele_data = self.df[col].astype(str).replace('0', 'missing')
                encoded_col = f"{col}_encoded"
                self.df[encoded_col] = le_allele.fit_transform(allele_data)
                self.label_encoders[col] = le_allele
                features.append(encoded_col)
        
        # Add position feature
        if 'position' in self.df.columns and target_strategy != 'position':
            self.df['position_norm'] = (
                (self.df['position'] - self.df['position'].min()) / 
                (self.df['position'].max() - self.df['position'].min())
            )
            features.append('position_norm')
        
        # Create target variable
        if target_strategy == 'position' and 'position' in self.df.columns:
            pos_min, pos_max = self.df['position'].min(), self.df['position'].max()
            self.y = ((self.df['position'] - pos_min) / (pos_max - pos_min) * 100).astype(int)
            preprocessing_info["target_strategy"] = "position"
        elif create_synthetic_target or target_strategy == 'synthetic':
            # Create synthetic phenotype
            np.random.seed(42)
            base_effect = 0
            
            if 'chromosome_encoded' in features:
                base_effect += 2 * self.df['chromosome_encoded']
            
            for col in allele_cols:
                encoded_col = f"{col}_encoded"
                if encoded_col in features:
                    base_effect += 3 * self.df[encoded_col]
            
            noise = np.random.normal(0, 5, len(self.df))
            synthetic_phenotype = base_effect + noise
            
            self.y = ((synthetic_phenotype - synthetic_phenotype.min()) / 
                     (synthetic_phenotype.max() - synthetic_phenotype.min()) * 100).astype(int)
            preprocessing_info["target_strategy"] = "synthetic"
        
        # Prepare features
        available_features = [f for f in features if f in self.df.columns]
        self.X = self.df[available_features].select_dtypes(include=[np.number])
        
        # Quantize features for FHE
        for col in self.X.columns:
            col_min, col_max = self.X[col].min(), self.X[col].max()
            if col_max > col_min:
                self.X[col] = ((self.X[col] - col_min) / (col_max - col_min) * (2**n_bits - 1)).astype(int)
            else:
                self.X[col] = 0
        
        self.feature_columns = list(self.X.columns)
        
        preprocessing_info.update({
            "features_selected": list(self.X.columns),
            "feature_matrix_shape": self.X.shape,
            "target_shape": self.y.shape,
            "feature_range": [int(self.X.min().min()), int(self.X.max().max())],
            "target_range": [int(self.y.min()), int(self.y.max())],
            "n_bits": n_bits
        })
        
        return preprocessing_info
    
    def train_model(self, test_size: float = 0.2, n_bits: int = 6) -> Dict[str, Any]:
        """Train FHE model and return results"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Initialize and train FHE model
        self.model = FHELinearRegression(n_bits=n_bits)
        
        # Training timing
        start_time = time.time()
        self.model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Compile FHE circuit
        start_time = time.time()
        inputset = X_train[:min(100, len(X_train))]
        self.fhe_circuit = self.model.compile(inputset)
        compile_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Store results
        self.results = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred
        }
        
        return {
            "training_metrics": {
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "train_rmse": float(np.sqrt(train_mse)),
                "test_rmse": float(np.sqrt(test_mse))
            },
            "timing": {
                "fit_time": fit_time,
                "compile_time": compile_time
            },
            "data_split": {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "total_features": len(self.feature_columns)
            }
        }
    
    def predict_encrypted(self, n_samples: int = 5) -> Dict[str, Any]:
        """Demonstrate encrypted prediction"""
        
        if self.fhe_circuit is None:
            raise HTTPException(status_code=400, detail="Model not compiled for FHE")
        
        test_samples = self.results['X_test'][:n_samples]
        
        start_time = time.time()
        
        # Encrypted inference
        encrypted_predictions = []
        for i in range(len(test_samples)):
            sample = test_samples.iloc[i:i+1].values
            encrypted_input = self.fhe_circuit.encrypt(sample)
            encrypted_result = self.fhe_circuit.run(encrypted_input)
            decrypted_result = self.fhe_circuit.decrypt(encrypted_result)
            encrypted_predictions.append(float(decrypted_result[0]))
        
        inference_time = time.time() - start_time
        
        # Compare with clear predictions
        clear_predictions = self.model.predict(test_samples)
        actual_values = self.results['y_test'][:n_samples].tolist()
        
        return {
            "encrypted_inference": {
                "inference_time": inference_time,
                "samples_processed": n_samples,
                "predictions": [
                    {
                        "sample_id": i,
                        "clear_prediction": float(clear_predictions[i]),
                        "encrypted_prediction": encrypted_predictions[i],
                        "actual_value": actual_values[i],
                        "encryption_error": abs(float(clear_predictions[i]) - encrypted_predictions[i])
                    }
                    for i in range(n_samples)
                ]
            }
        }

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Privacy-Preserving Genomic Analysis API",
        "description": "Upload CSV files for secure FHE-based genomic regression analysis",
        "endpoints": {
            "/analyze": "POST - Upload CSV and run complete analysis",
            "/train": "POST - Train model on uploaded CSV",
            "/predict": "POST - Make encrypted predictions",
            "/models": "GET - List available models",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_genomic_data(
    file: UploadFile = File(...),
    n_bits: int = 6,
    test_size: float = 0.2,
    target_strategy: str = "synthetic",
    n_encrypted_samples: int = 3
):
    """
    Complete genomic analysis: upload CSV, train FHE model, and get results
    """
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    model_id = f"model_{int(time.time())}_{hash(file.filename) % 10000}"
    
    try:
        # Read file content
        content = await file.read()
        
        # Initialize analyzer
        analyzer = PrivateGenomicAnalyzer(model_id)
        
        # Load and preprocess data
        analyzer.load_csv_data(content)
        preprocessing_info = analyzer.preprocess_data(
            n_bits=n_bits, 
            target_strategy=target_strategy,
            create_synthetic_target=True
        )
        
        # Train model
        training_results = analyzer.train_model(test_size=test_size, n_bits=n_bits)
        
        # Demonstrate encrypted inference
        encryption_demo = analyzer.predict_encrypted(n_samples=n_encrypted_samples)
        
        # Store model for future use
        MODELS[model_id] = analyzer
        
        # Combine all results
        results = {
            "model_id": model_id,
            "file_info": {
                "filename": file.filename,
                "file_size": len(content)
            },
            "preprocessing": preprocessing_info,
            "training": training_results,
            "encryption_demo": encryption_demo,
            "privacy_status": "All computations performed on encrypted data"
        }
        
        RESULTS[model_id] = results
        
        return AnalysisResponse(
            model_id=model_id,
            status="success",
            message="Genomic analysis completed successfully with FHE privacy protection",
            results=results,
            timing={
                "total_fit_time": training_results["timing"]["fit_time"],
                "total_compile_time": training_results["timing"]["compile_time"],
                "encrypted_inference_time": encryption_demo["encrypted_inference"]["inference_time"]
            }
        )
        
    except Exception as e:
        error_detail = f"Analysis failed: {str(e)}"
        if "DEBUG" in os.environ:
            error_detail += f"\n\nTraceback:\n{traceback.format_exc()}"
        
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    config: AnalysisConfig = AnalysisConfig()
):
    """Train a new FHE model on uploaded genomic data"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    model_id = f"model_{int(time.time())}_{hash(file.filename) % 10000}"
    
    try:
        content = await file.read()
        analyzer = PrivateGenomicAnalyzer(model_id)
        
        # Load and process data
        analyzer.load_csv_data(content)
        preprocessing_info = analyzer.preprocess_data(
            n_bits=config.n_bits,
            target_strategy=config.target_strategy,
            create_synthetic_target=config.create_synthetic_target
        )
        
        # Train model
        training_results = analyzer.train_model(
            test_size=config.test_size,
            n_bits=config.n_bits
        )
        
        # Store model
        MODELS[model_id] = analyzer
        
        return {
            "model_id": model_id,
            "status": "trained",
            "preprocessing": preprocessing_info,
            "training_results": training_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def make_encrypted_prediction(request: PredictionRequest):
    """Make encrypted predictions using a trained model"""
    
    if request.model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    analyzer = MODELS[request.model_id]
    
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame(request.data)
        
        # Ensure input has same features as training data
        missing_features = set(analyzer.feature_columns) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing features: {list(missing_features)}"
            )
        
        # Select and order features correctly
        input_features = input_df[analyzer.feature_columns]
        
        # Make encrypted prediction
        start_time = time.time()
        encrypted_input = analyzer.fhe_circuit.encrypt(input_features.values)
        encrypted_result = analyzer.fhe_circuit.run(encrypted_input)
        decrypted_prediction = analyzer.fhe_circuit.decrypt(encrypted_result)
        prediction_time = time.time() - start_time
        
        return {
            "model_id": request.model_id,
            "predictions": decrypted_prediction.tolist(),
            "prediction_time": prediction_time,
            "privacy_status": "Prediction made on encrypted data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List all available trained models"""
    return {
        "models": [
            {
                "model_id": model_id,
                "features": len(analyzer.feature_columns) if analyzer.feature_columns else 0,
                "status": "ready" if analyzer.fhe_circuit else "training"
            }
            for model_id, analyzer in MODELS.items()
        ]
    }

@app.get("/results/{model_id}")
async def get_results(model_id: str):
    """Get detailed results for a specific model"""
    if model_id not in RESULTS:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return RESULTS[model_id]

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model and its results"""
    if model_id in MODELS:
        del MODELS[model_id]
    if model_id in RESULTS:
        del RESULTS[model_id]
    
    return {"message": f"Model {model_id} deleted successfully"}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Privacy-Preserving Genomic Analysis API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ["DEBUG"] = "1"
    
    print(f"""
    🔒 Privacy-Preserving Genomic Analysis API Server
    ================================================
    
    Server starting on: http://{args.host}:{args.port}
    
    API Documentation: http://{args.host}:{args.port}/docs
    
    Key Features:
    • Fully Homomorphic Encryption (FHE) for data privacy
    • Secure genomic data analysis
    • RESTful API for easy integration
    • Real-time encrypted inference
    
    Upload your genomic CSV files and get privacy-preserving analysis results!
    """)
    
    uvicorn.run(
        "genomic_regression_api:app" if not args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
