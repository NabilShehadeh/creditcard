"""
Data processing utilities for credit card fraud detection.

This module handles data loading, preprocessing, and validation for the fraud detection system.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, preprocessing, and validation."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load credit card fraud dataset.
        
        Args:
            file_path: Path to CSV file. If None, uses sample data.
            
        Returns:
            DataFrame with transaction data
        """
        if file_path is None:
            # Generate sample data for demonstration
            return self._generate_sample_data()
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning(f"File {file_path} not found. Using sample data.")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate sample credit card transaction data for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample transaction data
        """
        np.random.seed(42)
        
        # Generate base features
        data = {
            'Time': np.random.exponential(1000, n_samples),
            'Amount': np.random.lognormal(3, 1.5, n_samples),
            'Class': np.random.binomial(1, 0.0017, n_samples)  # ~0.17% fraud rate
        }
        
        # Generate V1-V28 features (anonymized PCA components)
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame(data)
        
        # Add some realistic patterns for fraud cases
        fraud_mask = df['Class'] == 1
        if fraud_mask.sum() > 0:
            # Fraud transactions tend to have higher amounts
            df.loc[fraud_mask, 'Amount'] *= np.random.uniform(2, 5, fraud_mask.sum())
            
            # Some V features have different distributions for fraud
            df.loc[fraud_mask, 'V1'] += np.random.normal(2, 0.5, fraud_mask.sum())
            df.loc[fraud_mask, 'V3'] += np.random.normal(-1, 0.5, fraud_mask.sum())
            df.loc[fraud_mask, 'V4'] += np.random.normal(1.5, 0.5, fraud_mask.sum())
        
        logger.info(f"Generated sample data with {fraud_mask.sum()} fraud cases ({fraud_mask.sum()/len(df)*100:.2f}%)")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data for modeling.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Scale numerical features
        df_processed = self._scale_features(df_processed)
        
        # Add derived features
        df_processed = self._add_derived_features(df_processed)
        
        logger.info(f"Preprocessed data shape: {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For this dataset, missing values are rare, but we'll handle them
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            # Fill missing values with median for numerical columns
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using RobustScaler."""
        numerical_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        
        # Use RobustScaler to handle outliers
        self.scaler = RobustScaler()
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        logger.info("Scaled numerical features using RobustScaler")
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better fraud detection."""
        # Time-based features
        df['hour'] = (df['Time'] / 3600) % 24
        df['day_of_week'] = (df['Time'] / (3600 * 24)) % 7
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Interaction features
        df['time_amount_interaction'] = df['Time'] * df['Amount']
        
        # Statistical features from V columns
        v_cols = [f'V{i}' for i in range(1, 29)]
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_sum'] = df[v_cols].sum(axis=1)
        
        logger.info("Added derived features")
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   stratify_col: str = 'Class') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with time-based splitting.
        
        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing
            stratify_col: Column to stratify on
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Sort by time to ensure temporal order
        df_sorted = df.sort_values('Time').reset_index(drop=True)
        
        # Time-based split (last 20% for testing)
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"Split data: Train {len(train_df)}, Test {len(test_df)}")
        logger.info(f"Train fraud rate: {train_df['Class'].mean():.4f}")
        logger.info(f"Test fraud rate: {test_df['Class'].mean():.4f}")
        
        return train_df, test_df
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Preprocessed DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Define feature columns (exclude target and ID columns)
        exclude_cols = [target_col, 'Time']  # Time is used for temporal features but not as input
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared features: {len(feature_cols)} features")
        return X, y
    
    def save_preprocessor(self, file_path: str):
        """Save the fitted preprocessor."""
        if self.scaler is not None:
            joblib.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, file_path)
            logger.info(f"Saved preprocessor to {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """Load a fitted preprocessor."""
        try:
            preprocessor = joblib.load(file_path)
            self.scaler = preprocessor['scaler']
            self.feature_columns = preprocessor['feature_columns']
            logger.info(f"Loaded preprocessor from {file_path}")
        except FileNotFoundError:
            logger.warning(f"Preprocessor file {file_path} not found")


def load_and_preprocess_data(data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and preprocess data.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    processor = DataProcessor()
    
    # Load data
    df = processor.load_data(data_path)
    
    # Preprocess data
    df_processed = processor.preprocess_data(df)
    
    # Split data
    train_df, test_df = processor.split_data(df_processed)
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data()
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Train fraud rate: {train_df['Class'].mean():.4f}")
    print(f"Test fraud rate: {test_df['Class'].mean():.4f}")
