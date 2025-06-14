import requests
import pandas as pd
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path

class GeneticRegressionClient:
    """
    Client for interacting with the Genetic Data Linear Regression API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: The base URL of the FastAPI server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def train_and_save_model(
        self, 
        csv_file_path: str,
        target_column: Optional[str] = None,
        generate_target: bool = True,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload genetic data, train model, and save it as pickle.
        
        Args:
            csv_file_path: Path to the CSV file with genetic data
            target_column: Name of target column if present in data
            generate_target: Whether to generate synthetic target data
            model_name: Custom name for the saved model
            
        Returns:
            Dictionary with training results and model info
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Prepare the files and data
        files = {
            'file': ('genetic_data.csv', open(csv_file_path, 'rb'), 'text/csv')
        }
        
        data = {
            'generate_target': generate_target
        }
        
        if target_column:
            data['target_column'] = target_column
        
        if model_name:
            data['model_name'] = model_name
        
        try:
            response = self.session.post(
                f"{self.base_url}/train-and-save-model/",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
        
        finally:
            files['file'][1].close()
    
    def predict_with_saved_model(
        self, 
        csv_file_path: str, 
        model_filename: str
    ) -> Dict[str, Any]:
        """
        Make predictions using a previously saved model.
        
        Args:
            csv_file_path: Path to CSV file with genetic data for prediction
            model_filename: Name of the saved model file
            
        Returns:
            Dictionary with predictions and model info
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        files = {
            'file': ('prediction_data.csv', open(csv_file_path, 'rb'), 'text/csv')
        }
        
        data = {
            'model_filename': model_filename
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict-with-saved-model/",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
        
        finally:
            files['file'][1].close()
    
    def list_saved_models(self) -> Dict[str, Any]:
        """
        Get a list of all saved models.
        
        Returns:
            Dictionary with list of saved models and their metadata
        """
        try:
            response = self.session.get(f"{self.base_url}/list-models/")
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def download_model(self, model_filename: str, save_path: str = None) -> str:
        """
        Download a saved model file.
        
        Args:
            model_filename: Name of the model file to download
            save_path: Local path to save the file (optional)
            
        Returns:
            Path where the file was saved
        """
        if save_path is None:
            save_path = model_filename
        
        try:
            response = self.session.get(
                f"{self.base_url}/download-model/{model_filename}",
                stream=True
            )
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return save_path
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download model: {str(e)}")
    
    def create_sample_genetic_data(self, filename: str = "sample_genetic_data.csv", num_rows: int = 100):
        """
        Create a sample genetic data CSV file for testing.
        
        Args:
            filename: Name of the CSV file to create
            num_rows: Number of rows to generate
        """
        import random
        
        # Sample data generation
        chromosomes = list(range(1, 23)) + ['X', 'Y']
        alleles = ['A', 'T', 'G', 'C', '0']  # 0 represents missing
        
        data = []
        for i in range(num_rows):
            row = {
                'rsid': f'rs{random.randint(1000000, 9999999)}',
                'chromosome': random.choice(chromosomes),
                'position': random.randint(10000, 250000000),
                'allele1': random.choice(alleles),
                'allele2': random.choice(alleles)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Sample genetic data saved to: {filename}")
        return filename


def main():
    """
    Example usage of the GeneticRegressionClient
    """
    # Initialize client
    client = GeneticRegressionClient()
    
    # Check if server is running
    print("1. Checking server health...")
    health = client.health_check()
    print(f"Health check: {health}")
    
    if health.get('status') != 'healthy':
        print("Server is not running. Please start the FastAPI server first.")
        return
    
    # Create sample data
    print("\n2. Creating sample genetic data...")
    sample_file = client.create_sample_genetic_data("test_genetic_data.csv", num_rows=50)
    
    # Train and save model
    print("\n3. Training and saving model...")
    train_result = client.train_and_save_model(
        csv_file_path=sample_file,
        generate_target=True,
        model_name="test_model"
    )
    
    print("Training result:")
    print(json.dumps(train_result, indent=2))
    
    if train_result.get('status') == 'success':
        model_filename = train_result['model_file']['filename']
        print(f"\nModel saved as: {model_filename}")
        
        # List all models
        print("\n4. Listing all saved models...")
        models_list = client.list_saved_models()
        print("Available models:")
        for model in models_list.get('models', []):
            print(f"  - {model['filename']} (R²: {model.get('test_r2', 'N/A')})")
        
        # Make predictions with saved model
        print("\n5. Making predictions with saved model...")
        prediction_result = client.predict_with_saved_model(
            csv_file_path=sample_file,
            model_filename=model_filename
        )
        
        print("Prediction result:")
        print(f"  Status: {prediction_result.get('status')}")
        print(f"  Sample count: {prediction_result.get('sample_count')}")
        print(f"  First 5 predictions: {prediction_result.get('predictions', [])[:5]}")
        
        # Download model
        print("\n6. Downloading model...")
        try:
            download_path = client.download_model(model_filename, f"downloaded_{model_filename}")
            print(f"Model downloaded to: {download_path}")
        except Exception as e:
            print(f"Download failed: {e}")
    
    else:
        print("Training failed!")
    
    # Cleanup
    try:
        os.remove(sample_file)
        print(f"\nCleaned up sample file: {sample_file}")
    except:
        pass


if __name__ == "__main__":
    main()
