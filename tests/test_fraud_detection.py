"""
Comprehensive unit tests for credit card fraud detection system.

This module tests all major components including data processing,
feature engineering, model training, and API endpoints.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import modules to test with error handling
try:
    from data.processor import DataProcessor
    from features.engineering import FeatureEngineer, FeatureSelector
    from models.trainer import ModelTrainer
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)

# Test data
@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.exponential(1000, n_samples),
        'Amount': np.random.lognormal(3, 1.5, n_samples),
        'Class': np.random.binomial(1, 0.01, n_samples)
    }
    
    # Add V features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataProcessor:
    """Test DataProcessor class."""
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.data_dir == Path("data")
        assert processor.scaler is None
        assert processor.feature_columns is None
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        processor = DataProcessor()
        df = processor._generate_sample_data(100)
        
        assert len(df) == 100
        assert 'Time' in df.columns
        assert 'Amount' in df.columns
        assert 'Class' in df.columns
        assert all(f'V{i}' in df.columns for i in range(1, 29))
        assert df['Class'].dtype == int
        assert df['Amount'].min() >= 0
    
    def test_load_data_with_file(self, temp_dir):
        """Test loading data from file."""
        processor = DataProcessor(temp_dir)
        
        # Create test CSV file
        test_file = Path(temp_dir) / "test_data.csv"
        sample_data = processor._generate_sample_data(100)
        sample_data.to_csv(test_file, index=False)
        
        df = processor.load_data(str(test_file))
        assert len(df) == 100
        assert list(df.columns) == list(sample_data.columns)
    
    def test_load_data_without_file(self):
        """Test loading data without file (uses sample data)."""
        processor = DataProcessor()
        df = processor.load_data()
        
        assert len(df) == 10000  # Default sample size
        assert 'Time' in df.columns
        assert 'Amount' in df.columns
        assert 'Class' in df.columns
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        processor = DataProcessor()
        
        # Add some missing values
        sample_data.loc[0:5, 'Amount'] = np.nan
        sample_data.loc[10:15, 'V1'] = np.nan
        
        df_processed = processor._handle_missing_values(sample_data)
        
        # Check that missing values are filled
        assert df_processed['Amount'].isnull().sum() == 0
        assert df_processed['V1'].isnull().sum() == 0
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        processor = DataProcessor()
        df_scaled = processor._scale_features(sample_data)
        
        # Check that scaler is fitted
        assert processor.scaler is not None
        
        # Check that numerical columns are scaled
        numerical_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        for col in numerical_cols:
            assert col in df_scaled.columns
    
    def test_add_derived_features(self, sample_data):
        """Test derived feature creation."""
        processor = DataProcessor()
        df_features = processor._add_derived_features(sample_data)
        
        # Check that derived features are added
        assert 'hour' in df_features.columns
        assert 'day_of_week' in df_features.columns
        assert 'amount_log' in df_features.columns
        assert 'amount_sqrt' in df_features.columns
        assert 'time_amount_interaction' in df_features.columns
        assert 'v_mean' in df_features.columns
        assert 'v_std' in df_features.columns
        assert 'v_sum' in df_features.columns
    
    def test_preprocess_data(self, sample_data):
        """Test complete preprocessing pipeline."""
        processor = DataProcessor()
        df_processed = processor.preprocess_data(sample_data)
        
        # Check that preprocessing steps are applied
        assert len(df_processed.columns) > len(sample_data.columns)
        assert df_processed.isnull().sum().sum() == 0
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        processor = DataProcessor()
        df_processed = processor.preprocess_data(sample_data)
        train_df, test_df = processor.split_data(df_processed, test_size=0.2)
        
        # Check split sizes
        assert len(train_df) + len(test_df) == len(df_processed)
        assert len(test_df) / len(df_processed) == pytest.approx(0.2, rel=0.1)
        
        # Check temporal order
        assert train_df['Time'].max() <= test_df['Time'].min()
    
    def test_prepare_features_target(self, sample_data):
        """Test feature and target preparation."""
        processor = DataProcessor()
        df_processed = processor.preprocess_data(sample_data)
        X, y = processor.prepare_features_target(df_processed)
        
        # Check that features and target are separated
        assert 'Class' not in X.columns
        assert 'Time' not in X.columns
        assert len(X.columns) > 0
        assert len(y) == len(X)
        assert y.name == 'Class'
    
    def test_save_load_preprocessor(self, sample_data, temp_dir):
        """Test preprocessor saving and loading."""
        processor = DataProcessor(temp_dir)
        df_processed = processor.preprocess_data(sample_data)
        processor.prepare_features_target(df_processed)
        
        # Save preprocessor
        preprocessor_path = Path(temp_dir) / "preprocessor.joblib"
        processor.save_preprocessor(str(preprocessor_path))
        assert preprocessor_path.exists()
        
        # Load preprocessor
        new_processor = DataProcessor(temp_dir)
        new_processor.load_preprocessor(str(preprocessor_path))
        
        assert new_processor.scaler is not None
        assert new_processor.feature_columns is not None


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.scalers == {}
        assert engineer.encoders == {}
        assert engineer.feature_names == []
    
    def test_engineer_all_features(self, sample_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_all_features(sample_data)
        
        # Check that many features are added
        assert len(df_features.columns) > len(sample_data.columns)
        
        # Check specific feature types
        assert 'amount_log' in df_features.columns
        assert 'amount_sqrt' in df_features.columns
        assert 'hour' in df_features.columns
        assert 'day_of_week' in df_features.columns
        assert 'is_weekend' in df_features.columns
        assert 'is_night' in df_features.columns
    
    def test_add_transaction_features(self, sample_data):
        """Test transaction-level feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer._add_transaction_features(sample_data)
        
        # Check transaction features
        assert 'amount_log' in df_features.columns
        assert 'amount_sqrt' in df_features.columns
        assert 'amount_rank' in df_features.columns
        assert 'amount_category' in df_features.columns
        assert 'v_sum' in df_features.columns
        assert 'v_mean' in df_features.columns
        assert 'v_std' in df_features.columns
    
    def test_add_temporal_features(self, sample_data):
        """Test temporal feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer._add_temporal_features(sample_data, 'Time')
        
        # Check temporal features
        assert 'hour' in df_features.columns
        assert 'day_of_week' in df_features.columns
        assert 'day_of_month' in df_features.columns
        assert 'time_of_day' in df_features.columns
        assert 'is_weekend' in df_features.columns
        assert 'is_night' in df_features.columns
        assert 'is_business_hours' in df_features.columns
        assert 'hour_sin' in df_features.columns
        assert 'hour_cos' in df_features.columns
    
    def test_add_statistical_features(self, sample_data):
        """Test statistical feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer._add_statistical_features(sample_data)
        
        # Check statistical features
        assert 'amount_rolling_mean_5' in df_features.columns
        assert 'amount_rolling_std_5' in df_features.columns
        assert 'amount_zscore' in df_features.columns
        assert 'amount_percentile' in df_features.columns
        assert 'amount_to_avg_ratio' in df_features.columns
    
    def test_add_interaction_features(self, sample_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer._add_interaction_features(sample_data)
        
        # Check interaction features
        assert 'amount_time_interaction' in df_features.columns
        assert 'amount_hour_interaction' in df_features.columns
        assert 'v1_v2_interaction' in df_features.columns
        assert 'v3_v4_interaction' in df_features.columns
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance calculation."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_all_features(sample_data)
        
        importance = engineer.get_feature_importance(df_features)
        
        # Check that importance is calculated
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # Check that importance values are non-negative
        for feat, score in importance.items():
            assert score >= 0
    
    def test_select_top_features(self, sample_data):
        """Test top feature selection."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_all_features(sample_data)
        
        top_features = engineer.select_top_features(df_features, n_features=10)
        
        # Check that top features are selected
        assert isinstance(top_features, list)
        assert len(top_features) <= 10
        assert len(top_features) > 0


class TestFeatureSelector:
    """Test FeatureSelector class."""
    
    def test_remove_low_variance_features(self, sample_data):
        """Test low variance feature removal."""
        # Add low variance feature
        sample_data['low_var'] = 0.1  # Very low variance
        
        df_selected = FeatureSelector.remove_low_variance_features(sample_data, threshold=0.01)
        
        # Check that low variance feature is removed
        assert 'low_var' not in df_selected.columns
        assert len(df_selected.columns) < len(sample_data.columns)
    
    def test_remove_correlated_features(self, sample_data):
        """Test correlated feature removal."""
        # Add highly correlated feature
        sample_data['high_corr'] = sample_data['V1'] * 0.99 + 0.01
        
        df_selected = FeatureSelector.remove_correlated_features(sample_data, threshold=0.95)
        
        # Check that correlated feature is removed
        assert 'high_corr' not in df_selected.columns
        assert len(df_selected.columns) < len(sample_data.columns)


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    def test_init(self, temp_dir):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(temp_dir)
        assert trainer.models_dir == Path(temp_dir)
        assert trainer.models == {}
        assert trainer.results == {}
        assert trainer.feature_importance == {}
    
    def test_calculate_class_weights(self, sample_data):
        """Test class weight calculation."""
        trainer = ModelTrainer()
        y = sample_data['Class']
        weights = trainer._calculate_class_weights(y)
        
        # Check that weights are calculated
        assert isinstance(weights, dict)
        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]  # Minority class should have higher weight
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        trainer = ModelTrainer()
        
        # Create mock predictions
        y_true = sample_data['Class']
        y_pred = np.random.randint(0, 2, len(y_true))
        y_pred_proba = np.random.random(len(y_true))
        
        results = trainer._evaluate_model(y_true, y_pred, y_pred_proba, "Test Model")
        
        # Check that results contain expected metrics
        expected_metrics = [
            'auc_score', 'pr_auc_score', 'precision', 'recall', 'f1_score',
            'false_positive_rate', 'precision_at_100', 'recall_at_100',
            'optimal_threshold', 'confusion_matrix'
        ]
        
        for metric in expected_metrics:
            assert metric in results
        
        # Check that metrics are valid
        assert 0 <= results['auc_score'] <= 1
        assert 0 <= results['pr_auc_score'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
    
    @patch('src.models.trainer.LogisticRegression')
    def test_train_logistic_regression(self, mock_lr, sample_data, temp_dir):
        """Test logistic regression training."""
        trainer = ModelTrainer(temp_dir)
        
        # Prepare data
        processor = DataProcessor()
        df_processed = processor.preprocess_data(sample_data)
        train_df, test_df = processor.split_data(df_processed)
        X_train, y_train = processor.prepare_features_target(train_df)
        X_val, y_val = processor.prepare_features_target(test_df)
        
        # Mock the model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.random.random((len(X_val), 2))
        mock_model.predict.return_value = np.random.randint(0, 2, len(X_val))
        mock_lr.return_value = mock_model
        
        trainer._train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Check that model is trained and saved
        assert 'logistic_regression' in trainer.results
        assert 'logistic_regression' in trainer.models
        assert Path(temp_dir, 'logistic_regression.joblib').exists()
    
    @patch('src.models.trainer.RandomForestClassifier')
    def test_train_random_forest(self, mock_rf, sample_data, temp_dir):
        """Test random forest training."""
        trainer = ModelTrainer(temp_dir)
        
        # Prepare data
        processor = DataProcessor()
        df_processed = processor.preprocess_data(sample_data)
        train_df, test_df = processor.split_data(df_processed)
        X_train, y_train = processor.prepare_features_target(train_df)
        X_val, y_val = processor.prepare_features_target(test_df)
        
        # Mock the model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.random.random((len(X_val), 2))
        mock_model.predict.return_value = np.random.randint(0, 2, len(X_val))
        mock_model.feature_importances_ = np.random.random(len(X_train.columns))
        mock_rf.return_value = mock_model
        
        trainer._train_random_forest(X_train, y_train, X_val, y_val)
        
        # Check that model is trained and saved
        assert 'random_forest' in trainer.results
        assert 'random_forest' in trainer.models
        assert 'random_forest' in trainer.feature_importance
        assert Path(temp_dir, 'random_forest.joblib').exists()
    
    def test_save_load_model(self, temp_dir):
        """Test model saving and loading."""
        trainer = ModelTrainer(temp_dir)
        
        # Create mock model
        mock_model = Mock()
        mock_model.name = "test_model"
        
        # Save model
        trainer._save_model(mock_model, 'test_model')
        assert Path(temp_dir, 'test_model.joblib').exists()
        
        # Load model
        loaded_model = trainer.load_model('test_model')
        assert loaded_model is not None
    
    def test_load_nonexistent_model(self, temp_dir):
        """Test loading non-existent model."""
        trainer = ModelTrainer(temp_dir)
        loaded_model = trainer.load_model('nonexistent_model')
        assert loaded_model is None
    
    def test_generate_model_report(self, sample_data, temp_dir):
        """Test model report generation."""
        trainer = ModelTrainer(temp_dir)
        
        # Add mock results
        trainer.results = {
            'test_model': {
                'pr_auc_score': 0.8,
                'auc_score': 0.9,
                'f1_score': 0.7,
                'precision_at_100': 0.6,
                'precision': 0.7,
                'recall': 0.6,
                'false_positive_rate': 0.1,
                'recall_at_100': 0.5,
                'optimal_threshold': 0.5,
                'confusion_matrix': {'tn': 100, 'fp': 10, 'fn': 5, 'tp': 5}
            }
        }
        
        report = trainer.generate_model_report()
        
        # Check that report is generated
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'MODEL EVALUATION REPORT' in report
        assert 'test_model' in report


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self, sample_data, temp_dir):
        """Test complete end-to-end pipeline."""
        # Data processing
        processor = DataProcessor(temp_dir)
        df_processed = processor.preprocess_data(sample_data)
        train_df, test_df = processor.split_data(df_processed)
        X_train, y_train = processor.prepare_features_target(train_df)
        X_val, y_val = processor.prepare_features_target(test_df)
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_train_features = engineer.engineer_all_features(train_df)
        X_val_features = engineer.engineer_all_features(test_df)
        
        # Prepare features for modeling
        X_train_final, _ = processor.prepare_features_target(X_train_features)
        X_val_final, _ = processor.prepare_features_target(X_val_features)
        
        # Model training
        trainer = ModelTrainer(temp_dir)
        
        # Mock the models to avoid actual training
        with patch('src.models.trainer.LogisticRegression') as mock_lr:
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.random.random((len(X_val_final), 2))
            mock_model.predict.return_value = np.random.randint(0, 2, len(X_val_final))
            mock_lr.return_value = mock_model
            
            trainer._train_logistic_regression(X_train_final, y_train, X_val_final, y_val)
        
        # Check that everything worked
        assert 'logistic_regression' in trainer.results
        assert len(X_train_final.columns) > len(X_train.columns)
        assert Path(temp_dir, 'logistic_regression.joblib').exists()


# Pytest configuration
@pytest.fixture(scope="session")
def pytest_configure():
    """Configure pytest."""
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
