"""
Dataset Downloader Utility for Firmus AI Factory

This module provides utilities to download and manage public datasets
for sensor emulation and model validation.
"""

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and manage public datasets for AI factory sensor emulation"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize dataset downloader
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset registry
        self.datasets = {
            "butter_e": {
                "name": "BUTTER-E Energy Consumption Dataset",
                "kaggle_id": "pavankumar4757/butter-e-energy-data-for-deep-learning-models",
                "files": ["BUTTER-E Energy.csv"],
                "description": "Energy consumption data from 63,527 DL training runs"
            },
            "ieee_server": {
                "name": "IEEE DataPort Server Energy Dataset",
                "url": "https://ieee-dataport.org/open-access/data-server-energy-consumption-dataset",
                "description": "Real-world server energy consumption telemetry"
            }
        }
    
    def download_butter_e_kaggle(self, output_dir: Optional[str] = None) -> Path:
        """
        Download BUTTER-E dataset using Kaggle API
        
        Requires: kaggle API credentials configured (~/.kaggle/kaggle.json)
        Install: pip install kaggle
        
        Args:
            output_dir: Optional output directory (defaults to self.data_dir)
            
        Returns:
            Path to downloaded dataset directory
        """
        try:
            import kaggle
        except ImportError:
            logger.error("Kaggle API not installed. Run: pip install kaggle")
            logger.info("Also configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
            raise
        
        output_path = Path(output_dir) if output_dir else self.data_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_id = self.datasets["butter_e"]["kaggle_id"]
        logger.info(f"Downloading BUTTER-E dataset from Kaggle: {dataset_id}")
        
        kaggle.api.dataset_download_files(
            dataset_id,
            path=str(output_path),
            unzip=True
        )
        
        logger.info(f"Dataset downloaded to: {output_path}")
        return output_path
    
    def download_butter_e_manual(self) -> str:
        """
        Provide manual download instructions for BUTTER-E dataset
        
        Returns:
            Instructions string
        """
        instructions = """
        Manual Download Instructions for BUTTER-E Dataset:
        
        Option 1: Kaggle (Recommended)
        1. Visit: https://www.kaggle.com/datasets/pavankumar4757/butter-e-energy-data-for-deep-learning-models
        2. Click "Download" button
        3. Extract ZIP file to: {data_dir}/butter_e/
        
        Option 2: OpenEI
        1. Visit: https://data.openei.org/submissions/5991
        2. Download "BUTTER-E Energy.zip"
        3. Extract to: {data_dir}/butter_e/
        
        Expected file: BUTTER-E Energy.csv
        """.format(data_dir=self.data_dir)
        
        logger.info(instructions)
        return instructions
    
    def load_butter_e(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load BUTTER-E dataset into pandas DataFrame
        
        Args:
            file_path: Optional path to CSV file (auto-detects if None)
            
        Returns:
            DataFrame with energy consumption data
        """
        if file_path is None:
            # Try to find the file
            possible_paths = [
                self.data_dir / "butter_e" / "BUTTER-E Energy.csv",
                self.data_dir / "BUTTER-E Energy.csv",
                Path("data/raw/butter_e/BUTTER-E Energy.csv")
            ]
            
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
            
            if file_path is None:
                raise FileNotFoundError(
                    f"BUTTER-E dataset not found. Please download manually:\n"
                    f"{self.download_butter_e_manual()}"
                )
        
        logger.info(f"Loading BUTTER-E dataset from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset
        
        Args:
            dataset_name: Name of dataset (e.g., 'butter_e')
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.datasets[dataset_name]
    
    def list_datasets(self) -> None:
        """Print available datasets"""
        logger.info("Available datasets:")
        for name, info in self.datasets.items():
            logger.info(f"  - {name}: {info['name']}")
            logger.info(f"    {info['description']}")


def download_sample_data() -> Path:
    """
    Convenience function to download sample data for testing
    
    Returns:
        Path to downloaded data directory
    """
    downloader = DatasetDownloader()
    
    try:
        # Try Kaggle API first
        return downloader.download_butter_e_kaggle()
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Please download manually:")
        downloader.download_butter_e_manual()
        raise


if __name__ == "__main__":
    # Example usage
    downloader = DatasetDownloader()
    downloader.list_datasets()
    
    # Attempt to load BUTTER-E dataset
    try:
        df = downloader.load_butter_e()
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:\n{df.head()}")
    except FileNotFoundError as e:
        print(f"\n{e}")
