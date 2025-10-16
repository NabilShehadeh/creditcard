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
        df_processed = df.copy()
        
        # Check for missing values
        missing_counts = df_processed.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill missing values with median for numerical columns
            for col in df_processed.select_dtypes(include=[np.number]).columns:
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {median_val}")
            
            # Fill missing values with mode for categorical columns
            for col in df_processed.select_dtypes(include=['object']).columns:
                if df_processed[col].isnull().sum() > 0:
                    mode_val = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        # Check for infinite values
        inf_counts = np.isinf(df_processed.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            logger.warning(f"Found infinite values: {inf_counts[inf_counts > 0].to_dict()}")
            # Replace infinite values with large finite values
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            # Fill the NaN values created by replacing inf with median
            for col in df_processed.select_dtypes(include=[np.number]).columns:
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
        
        # Final check - ensure no NaN values remain
        final_missing = df_processed.isnull().sum().sum()
        if final_missing > 0:
            logger.error(f"Still have {final_missing} missing values after processing!")
            # Drop any remaining rows with NaN values as last resort
            df_processed = df_processed.dropna()
            logger.warning(f"Dropped rows with NaN values. New shape: {df_processed.shape}")
        
        return df_processed
    
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
        df_processed = df.copy()
        
        # Time-based features
        df_processed['hour'] = (df_processed['Time'] / 3600) % 24
        df_processed['day_of_week'] = (df_processed['Time'] / (3600 * 24)) % 7
        
        # Amount-based features (handle negative amounts)
        df_processed['amount_log'] = np.log1p(np.abs(df_processed['Amount']))
        df_processed['amount_sqrt'] = np.sqrt(np.abs(df_processed['Amount']))
        
        # Interaction features
        df_processed['time_amount_interaction'] = df_processed['Time'] * df_processed['Amount']
        
        # Statistical features from V columns
        v_cols = [f'V{i}' for i in range(1, 29)]
        df_processed['v_mean'] = df_processed[v_cols].mean(axis=1)
        df_processed['v_std'] = df_processed[v_cols].std(axis=1)
        df_processed['v_sum'] = df_processed[v_cols].sum(axis=1)
        
        # Handle any NaN values that might have been created
        df_processed = self._handle_missing_values(df_processed)
        
        logger.info("Added derived features")
        return df_processed
    
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
        
        # Final validation - ensure no NaN values
        if X.isnull().sum().sum() > 0:
            logger.error(f"Found NaN values in features: {X.isnull().sum().sum()}")
            # Fill any remaining NaN values
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if X[col].dtype in ['object']:
                        X[col].fillna('unknown', inplace=True)
                    else:
                        X[col].fillna(X[col].median(), inplace=True)
                    logger.warning(f"Filled NaN values in {col}")
        
        if y.isnull().sum() > 0:
            logger.error(f"Found NaN values in target: {y.isnull().sum()}")
            # Drop rows with NaN target values
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
            logger.warning(f"Dropped {sum(~valid_mask)} rows with NaN target values")
        
        # Final check
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            logger.error("Still have NaN values after final processing!")
            raise ValueError("Cannot proceed with NaN values in features or target")
        
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")
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
