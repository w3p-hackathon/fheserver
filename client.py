#!/usr/bin/env python3
"""
Client script for Privacy-Preserving Genomic Analysis API
Demonstrates how to interact with the FHE genomic regression API
"""

import requests
import json
import time
from pathlib import Path
import argparse

class GenomicAnalysisClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """Check if the API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def analyze_file(self, file_path: str, n_bits: int = 6, test_size: float = 0.2, 
                     n_encrypted_samples: int = 3):
        """
        Upload CSV file and perform complete genomic analysis
        
        Args:
            file_path: Path to CSV file
            n_bits: FHE precision bits (6-8 recommended)
            test_size: Fraction of data for testing
            n_encrypted_samples: Number of samples for encrypted demo
        """
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"🔒 Uploading {file_path.name} for privacy-preserving analysis...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'text/csv')}
            params = {
                'n_bits': n_bits,
                'test_size': test_size,
                'n_encrypted_samples': n_encrypted_samples
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/analyze",
                files=files,
                params=params
            )
            total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed in {total_time:.2f} seconds")
            return result
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
            return None
    
    def train_model(self, file_path: str, config: dict = None):
        """Train a new model on uploaded data"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"🎯 Training model on {file_path.name}...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'text/csv')}
            
            # Send config as JSON if provided
            data = {}
            if config:
                data['config'] = json.dumps(config)
            
            response = self.session.post(
                f"{self.base_url}/train",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model trained successfully: {result['model_id']}")
            return result
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(response.text)
            return None
    
    def make_prediction(self, model_id: str, data: list):
        """Make encrypted prediction using trained model"""
        
        print(f"🔮 Making encrypted prediction with model {model_id}...")
        
        payload = {
            "model_id": model_id,
            "data": data
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction completed in {result['prediction_time']:.3f} seconds")
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            return None
    
    def list_models(self):
        """List all available models"""
        response = self.session.get(f"{self.base_url}/models")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return None
    
    def get_results(self, model_id: str):
        """Get detailed results for a model"""
        response = self.session.get(f"{self.base_url}/results/{model_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get results: {response.status_code}")
            return None
    
    def delete_model(self, model_id: str):
        """Delete a model"""
        response = self.session.delete(f"{self.base_url}/models/{model_id}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Failed to delete model: {response.status_code}")
            return False

def print_analysis_summary(result):
    """Print a nice summary of analysis results"""
    if not result or 'results' not in result:
        print("❌ No results to display")
        return
    
    results = result['results']
    
    print("\n" + "="*60)
    print("🔒 PRIVACY-PRESERVING GENOMIC ANALYSIS RESULTS")
    print("="*60)
    
    # File info
    if 'file_info' in results:
        file_info = results['file_info']
        print(f"📁 File: {file_info['filename']} ({file_info['file_size']} bytes)")
    
    # Preprocessing info
    if 'preprocessing' in results:
        prep = results['preprocessing']
        print(f"🧬 Samples: {prep['original_shape'][0]}")
        print(f"📊 Features: {len(prep['features_selected'])}")
        print(f"🎯 Target: {prep['target_strategy']}")
    
    # Training results
    if 'training' in results:
        training = results['training']
        metrics = training['training_metrics']
        timing = training['timing']
        
        print(f"\n📈 MODEL PERFORMANCE:")
        print(f"   Training R²: {metrics['train_r2']:.4f}")
        print(f"   Test R²:     {metrics['test_r2']:.4f}")
        print(f"   Test RMSE:   {metrics['test_rmse']:.4f}")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Training:    {timing['fit_time']:.2f}s")
        print(f"   Compilation: {timing['compile_time']:.2f}s")
    
    # Encryption demo
    if 'encryption_demo' in results:
        demo = results['encryption_demo']['encrypted_inference']
        print(f"\n🔐 ENCRYPTED INFERENCE DEMO:")
        print(f"   Samples:     {demo['samples_processed']}")
        print(f"   Time:        {demo['inference_time']:.3f}s")
        
        print(f"\n   Predictions (Clear vs Encrypted):")
        for pred in demo['predictions'][:3]:  # Show first 3
            clear = pred['clear_prediction']
            encrypted = pred['encrypted_prediction']
            error = pred['encryption_error']
            print(f"   Sample {pred['sample_id']+1}: {clear:.2f} vs {encrypted:.2f} (error: {error:.3f})")
    
    print(f"\n✅ Privacy Status: {results.get('privacy_status', 'Encrypted')}")
    print("="*60)

def create_sample_data():
    """Create a sample genomic CSV file for testing"""
    sample_data = """rsid,chromosome,position,allele1,allele2
rs4477212,1,82154,T,T
rs3131972,1,752721,G,G
rs12562034,1,768448,G,G
rs11240777,1,798959,G,G
rs6681049,1,800007,C,C
rs4970383,1,838555,A,A
rs4475691,1,846808,C,C
rs7537756,1,854250,A,A
rs13302982,1,861808,G,G
rs1110052,1,873558,T,T
rs17160698,1,887162,T,T
rs3748597,1,888659,C,C
rs13303106,1,891945,A,G
rs28415373,1,893981,C,C
rs13303010,1,894573,A,G
rs6696281,1,903104,T,C
rs28391282,1,904165,G,G
rs2340592,1,910935,G,G
rs13303118,1,918384,T,G
rs2341354,1,918573,A,G"""
    
    with open("sample_genomic_data.csv", "w") as f:
        f.write(sample_data)
    
    print("📄 Created sample_genomic_data.csv for testing")
    return "sample_genomic_data.csv"

def main():
    parser = argparse.ArgumentParser(description="Genomic Analysis API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--file", help="CSV file to analyze")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data file")
    parser.add_argument("--demo", action="store_true", help="Run complete demo")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    client = GenomicAnalysisClient(args.url)
    
    # Check server health
    if not client.health_check():
        print(f"❌ Cannot connect to API server at {args.url}")
        print("Make sure the server is running with: python genomic_regression_api.py")
        return
    
    print(f"✅ Connected to API server at {args.url}")
    
    if args.create_sample:
        create_sample_data()
        return
    
    if args.list_models:
        models = client.list_models()
        if models and models['models']:
            print("\n📋 Available Models:")
            for model in models['models']:
                print(f"   {model['model_id']} - {model['features']} features ({model['status']})")
        else:
            print("📋 No models available")
        return
    
    if args.demo:
        print("🚀 Running complete demo...")
        
        # Create sample data if it doesn't exist
        sample_file = "sample_genomic_data.csv"
        if not Path(sample_file).exists():
            sample_file = create_sample_data()
        
        # Run analysis
        result = client.analyze_file(sample_file, n_bits=6, n_encrypted_samples=3)
        if result:
            print_analysis_summary(result)
        
        return
    
    if args.file:
        if not Path(args.file).exists():
            print(f"❌ File not found: {args.file}")
            return
        
        # Analyze the specified file
        result = client.analyze_file(args.file)
        if result:
            print_analysis_summary(result)
    else:
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python api_client.py --demo          # Run demo with sample data")
        print("  python api_client.py --file data.csv # Analyze your CSV file")

if __name__ == "__main__":
    main()
