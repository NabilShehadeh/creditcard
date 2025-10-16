"""
Advanced feature engineering for credit card fraud detection.

This module implements sophisticated feature engineering techniques including
behavioral features, temporal patterns, and anomaly detection features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for fraud detection."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def engineer_all_features(self, df: pd.DataFrame, 
                            card_id_col: str = 'card_id',
                            time_col: str = 'Time') -> pd.DataFrame:
        """
        Engineer all types of features for fraud detection.
        
        Args:
            df: Input DataFrame
            card_id_col: Column name for card/account ID
            time_col: Column name for timestamp
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Add card_id if not present (for demonstration)
        if card_id_col not in df_features.columns:
            df_features[card_id_col] = np.random.randint(1, 1000, len(df_features))
        
        # 1. Transaction-level features
        df_features = self._add_transaction_features(df_features)
        
        # 2. Behavioral features (card-level)
        df_features = self._add_behavioral_features(df_features, card_id_col, time_col)
        
        # 3. Temporal features
        df_features = self._add_temporal_features(df_features, time_col)
        
        # 4. Statistical features
        df_features = self._add_statistical_features(df_features)
        
        # 5. Anomaly detection features
        df_features = self._add_anomaly_features(df_features)
        
        # 6. Interaction features
        df_features = self._add_interaction_features(df_features)
        
        logger.info(f"Engineered features. Final shape: {df_features.shape}")
        return df_features
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction-level features."""
        # Amount-based features
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        df['amount_rank'] = df['Amount'].rank(pct=True)
        
        # Amount categories
        df['amount_category'] = pd.cut(df['Amount'], 
                                     bins=[0, 10, 50, 100, 500, 1000, np.inf],
                                     labels=['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge'])
        
        # V-feature aggregations
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            df['v_sum'] = df[v_cols].sum(axis=1)
            df['v_mean'] = df[v_cols].mean(axis=1)
            df['v_std'] = df[v_cols].std(axis=1)
            df['v_max'] = df[v_cols].max(axis=1)
            df['v_min'] = df[v_cols].min(axis=1)
            df['v_range'] = df['v_max'] - df['v_min']
            
            # V-feature ratios
            df['v1_v2_ratio'] = df['V1'] / (df['V2'] + 1e-8)
            df['v3_v4_ratio'] = df['V3'] / (df['V4'] + 1e-8)
        
        logger.info("Added transaction-level features")
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame, 
                                card_id_col: str, time_col: str) -> pd.DataFrame:
        """Add behavioral features based on card history."""
        df_sorted = df.sort_values([card_id_col, time_col]).reset_index(drop=True)
        
        # Initialize behavioral features
        behavioral_features = [
            'count_tx_last_1h', 'count_tx_last_24h', 'count_tx_last_7d',
            'sum_amount_last_1h', 'sum_amount_last_24h', 'sum_amount_last_7d',
            'avg_amount_last_7d', 'std_amount_last_7d', 'max_amount_last_30d',
            'unique_merchants_last_30d', 'days_since_last_tx', 'tx_velocity'
        ]
        
        for feat in behavioral_features:
            df_sorted[feat] = 0.0
        
        # Calculate behavioral features for each card
        for card_id in df_sorted[card_id_col].unique():
            card_mask = df_sorted[card_id_col] == card_id
            card_data = df_sorted[card_mask].copy()
            
            if len(card_data) < 2:
                continue
                
            # Calculate rolling features
            for i in range(1, len(card_data)):
                current_time = card_data.iloc[i][time_col]
                current_amount = card_data.iloc[i]['Amount']
                
                # Time windows (in seconds)
                window_1h = current_time - 3600
                window_24h = current_time - 86400
                window_7d = current_time - 604800
                window_30d = current_time - 2592000
                
                # Previous transactions in each window
                prev_1h = card_data[card_data[time_col] >= window_1h].iloc[:-1]
                prev_24h = card_data[card_data[time_col] >= window_24h].iloc[:-1]
                prev_7d = card_data[card_data[time_col] >= window_7d].iloc[:-1]
                prev_30d = card_data[card_data[time_col] >= window_30d].iloc[:-1]
                
                # Count features
                df_sorted.loc[card_mask, 'count_tx_last_1h'].iloc[i] = len(prev_1h)
                df_sorted.loc[card_mask, 'count_tx_last_24h'].iloc[i] = len(prev_24h)
                df_sorted.loc[card_mask, 'count_tx_last_7d'].iloc[i] = len(prev_7d)
                
                # Amount features
                if len(prev_1h) > 0:
                    df_sorted.loc[card_mask, 'sum_amount_last_1h'].iloc[i] = prev_1h['Amount'].sum()
                if len(prev_24h) > 0:
                    df_sorted.loc[card_mask, 'sum_amount_last_24h'].iloc[i] = prev_24h['Amount'].sum()
                if len(prev_7d) > 0:
                    df_sorted.loc[card_mask, 'sum_amount_last_7d'].iloc[i] = prev_7d['Amount'].sum()
                    df_sorted.loc[card_mask, 'avg_amount_last_7d'].iloc[i] = prev_7d['Amount'].mean()
                    df_sorted.loc[card_mask, 'std_amount_last_7d'].iloc[i] = prev_7d['Amount'].std()
                
                if len(prev_30d) > 0:
                    df_sorted.loc[card_mask, 'max_amount_last_30d'].iloc[i] = prev_30d['Amount'].max()
                
                # Days since last transaction
                if len(prev_24h) > 0:
                    last_tx_time = prev_24h[time_col].max()
                    df_sorted.loc[card_mask, 'days_since_last_tx'].iloc[i] = (current_time - last_tx_time) / 86400
                
                # Transaction velocity
                if len(prev_24h) > 0 and len(prev_1h) > 0:
                    df_sorted.loc[card_mask, 'tx_velocity'].iloc[i] = len(prev_1h) / len(prev_24h)
        
        # Fill NaN values
        behavioral_cols = [col for col in df_sorted.columns if col in behavioral_features]
        df_sorted[behavioral_cols] = df_sorted[behavioral_cols].fillna(0)
        
        logger.info("Added behavioral features")
        return df_sorted
    
    def _add_temporal_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Add temporal features."""
        # Convert time to datetime-like features
        df['hour'] = (df[time_col] / 3600) % 24
        df['day_of_week'] = (df[time_col] / (3600 * 24)) % 7
        df['day_of_month'] = (df[time_col] / (3600 * 24)) % 30
        
        # Time buckets
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'])
        
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Added temporal features")
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Rolling statistics for amount
        df['amount_rolling_mean_5'] = df['Amount'].rolling(window=5, min_periods=1).mean()
        df['amount_rolling_std_5'] = df['Amount'].rolling(window=5, min_periods=1).std()
        df['amount_zscore'] = (df['Amount'] - df['amount_rolling_mean_5']) / (df['amount_rolling_std_5'] + 1e-8)
        
        # Percentile features
        df['amount_percentile'] = df['Amount'].rank(pct=True)
        
        # Ratio features
        df['amount_to_avg_ratio'] = df['Amount'] / (df['amount_rolling_mean_5'] + 1e-8)
        
        logger.info("Added statistical features")
        return df
    
    def _add_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly detection features."""
        # Isolation Forest features
        from sklearn.ensemble import IsolationForest
        
        # Use V-features for anomaly detection
        v_cols = [col for col in df.columns if col.startswith('V')]
        if len(v_cols) > 0:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df['isolation_score'] = iso_forest.fit_predict(df[v_cols])
            df['isolation_score'] = iso_forest.decision_function(df[v_cols])
        
        # PCA reconstruction error
        if len(v_cols) > 5:
            pca = PCA(n_components=min(10, len(v_cols)))
            v_data = df[v_cols].fillna(0)
            v_transformed = pca.fit_transform(v_data)
            v_reconstructed = pca.inverse_transform(v_transformed)
            reconstruction_error = np.mean((v_data - v_reconstructed) ** 2, axis=1)
            df['pca_reconstruction_error'] = reconstruction_error
        
        logger.info("Added anomaly detection features")
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""
        # Amount-time interactions
        df['amount_time_interaction'] = df['Amount'] * df['Time']
        df['amount_hour_interaction'] = df['Amount'] * df['hour']
        
        # Behavioral interactions
        if 'count_tx_last_24h' in df.columns:
            df['amount_velocity_interaction'] = df['Amount'] * df['count_tx_last_24h']
        
        # V-feature interactions
        v_cols = [col for col in df.columns if col.startswith('V')]
        if len(v_cols) >= 2:
            df['v1_v2_interaction'] = df['V1'] * df['V2']
            df['v3_v4_interaction'] = df['V3'] * df['V4']
        
        logger.info("Added interaction features")
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'Class') -> Dict[str, float]:
        """
        Calculate feature importance using correlation with target.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Dictionary of feature importance scores
        """
        feature_cols = [col for col in df.columns if col != target_col]
        correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        
        return correlations.to_dict()
    
    def select_top_features(self, df: pd.DataFrame, target_col: str = 'Class', 
                           n_features: int = 50) -> List[str]:
        """
        Select top features based on importance.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            n_features: Number of top features to select
            
        Returns:
            List of top feature names
        """
        importance = self.get_feature_importance(df, target_col)
        top_features = list(importance.keys())[:n_features]
        
        logger.info(f"Selected top {len(top_features)} features")
        return top_features


class FeatureSelector:
    """Feature selection utilities."""
    
    @staticmethod
    def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance."""
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        df_selected = selector.fit_transform(df)
        
        # Get selected feature names
        selected_features = df.columns[selector.get_support()]
        df_result = pd.DataFrame(df_selected, columns=selected_features, index=df.index)
        
        logger.info(f"Removed {len(df.columns) - len(selected_features)} low variance features")
        return df_result
    
    @staticmethod
    def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        df_result = df.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return df_result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.exponential(1000, n_samples),
        'Amount': np.random.lognormal(3, 1.5, n_samples),
        'Class': np.random.binomial(1, 0.01, n_samples)
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_all_features(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_features.columns)}")
    print(f"New features: {len(df_features.columns) - len(df.columns)}")
    
    # Get feature importance
    importance = engineer.get_feature_importance(df_features)
    print("\nTop 10 most important features:")
    for i, (feat, score) in enumerate(list(importance.items())[:10]):
        print(f"{i+1}. {feat}: {score:.4f}")
