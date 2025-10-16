"""
Comprehensive model training and evaluation for credit card fraud detection.

This module implements multiple ML models with proper evaluation metrics
and handles class imbalance through various techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Advanced ML
import lightgbm as lgb
import xgboost as xgb
# Import imbalanced learning modules with error handling
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. Some sampling methods will be disabled.")
    SMOTE = None
    RandomUnderSampler = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Comprehensive model training for fraud detection."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train all models and return results.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with model results
        """
        logger.info("Starting model training...")
        
        # 1. Baseline Logistic Regression
        self._train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # 2. Random Forest
        self._train_random_forest(X_train, y_train, X_val, y_val)
        
        # 3. LightGBM
        self._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # 4. XGBoost
        self._train_xgboost(X_train, y_train, X_val, y_val)
        
        # 5. Autoencoder (Unsupervised)
        self._train_autoencoder(X_train, y_train, X_val, y_val)
        
        # 6. Ensemble Model
        self._train_ensemble(X_train, y_train, X_val, y_val)
        
        logger.info("Model training completed!")
        return self.results
    
    def _train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series):
        """Train Logistic Regression with class weights."""
        logger.info("Training Logistic Regression...")
        
        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train)
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                class_weight=class_weights,
                random_state=42,
                max_iter=1000
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = pipeline.predict(X_val)
        
        results = self._evaluate_model(y_val, y_pred, y_pred_proba, "Logistic Regression")
        self.results['logistic_regression'] = results
        self.models['logistic_regression'] = pipeline
        
        # Save model
        self._save_model(pipeline, 'logistic_regression')
        
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series):
        """Train Random Forest with class weights."""
        logger.info("Training Random Forest...")
        
        class_weights = self._calculate_class_weights(y_train)
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        y_pred_proba = rf.predict_proba(X_val)[:, 1]
        y_pred = rf.predict(X_val)
        
        results = self._evaluate_model(y_val, y_pred, y_pred_proba, "Random Forest")
        self.results['random_forest'] = results
        self.models['random_forest'] = rf
        
        # Feature importance
        self.feature_importance['random_forest'] = dict(zip(
            X_train.columns, rf.feature_importances_
        ))
        
        self._save_model(rf, 'random_forest')
        
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series):
        """Train LightGBM with hyperparameter tuning."""
        logger.info("Training LightGBM...")
        
        # LightGBM parameters optimized for fraud detection
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predictions
        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results = self._evaluate_model(y_val, y_pred, y_pred_proba, "LightGBM")
        self.results['lightgbm'] = results
        self.models['lightgbm'] = model
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        self.feature_importance['lightgbm'] = dict(zip(X_train.columns, importance))
        
        self._save_model(model, 'lightgbm')
        
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series):
        """Train XGBoost with hyperparameter tuning."""
        logger.info("Training XGBoost...")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        results = self._evaluate_model(y_val, y_pred, y_pred_proba, "XGBoost")
        self.results['xgboost'] = results
        self.models['xgboost'] = model
        
        # Feature importance
        self.feature_importance['xgboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        self._save_model(model, 'xgboost')
        
    def _train_autoencoder(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series):
        """Train Autoencoder for unsupervised anomaly detection."""
        logger.info("Training Autoencoder...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create autoencoder
        input_dim = X_train_scaled.shape[1]
        encoding_dim = max(32, input_dim // 4)
        
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
        encoder = layers.Dense(encoding_dim // 2, activation="relu")(encoder)
        encoder = layers.Dense(encoding_dim // 4, activation="relu")(encoder)
        
        # Decoder
        decoder = layers.Dense(encoding_dim // 2, activation="relu")(encoder)
        decoder = layers.Dense(encoding_dim, activation="relu")(decoder)
        decoder = layers.Dense(input_dim, activation="sigmoid")(decoder)
        
        autoencoder = keras.Model(input_layer, decoder)
        
        # Compile and train
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train only on normal transactions
        normal_mask = y_train == 0
        X_normal = X_train_scaled[normal_mask]
        
        history = autoencoder.fit(
            X_normal, X_normal,
            epochs=50,
            batch_size=256,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate reconstruction error
        X_val_reconstructed = autoencoder.predict(X_val_scaled)
        reconstruction_error = np.mean((X_val_scaled - X_val_reconstructed) ** 2, axis=1)
        
        # Use reconstruction error as anomaly score
        # Higher error = more likely to be fraud
        y_pred_proba = reconstruction_error / np.max(reconstruction_error)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results = self._evaluate_model(y_val, y_pred, y_pred_proba, "Autoencoder")
        self.results['autoencoder'] = results
        self.models['autoencoder'] = {
            'model': autoencoder,
            'scaler': scaler
        }
        
        self._save_model({'model': autoencoder, 'scaler': scaler}, 'autoencoder')
        
    def _train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series):
        """Train ensemble model combining multiple approaches."""
        logger.info("Training Ensemble Model...")
        
        # Get predictions from all models
        ensemble_predictions = []
        
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                # Special handling for autoencoder
                scaler = model['scaler']
                X_val_scaled = scaler.transform(X_val)
                X_val_reconstructed = model['model'].predict(X_val_scaled)
                reconstruction_error = np.mean((X_val_scaled - X_val_reconstructed) ** 2, axis=1)
                pred_proba = reconstruction_error / np.max(reconstruction_error)
            elif model_name == 'lightgbm':
                pred_proba = model.predict(X_val)
            else:
                pred_proba = model.predict_proba(X_val)[:, 1]
            
            ensemble_predictions.append(pred_proba)
        
        # Simple averaging ensemble
        ensemble_pred_proba = np.mean(ensemble_predictions, axis=0)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        results = self._evaluate_model(y_val, ensemble_pred, ensemble_pred_proba, "Ensemble")
        self.results['ensemble'] = results
        
        # Store ensemble weights (equal weights for now)
        self.models['ensemble'] = {
            'weights': [1.0 / len(ensemble_predictions)] * len(ensemble_predictions),
            'models': list(self.models.keys())
        }
        
    def _calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def _evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate model performance."""
        # Basic metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        pr_auc_score = average_precision_score(y_true, y_pred_proba)
        
        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find optimal threshold (maximize F1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Predictions with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
        
        # Calculate metrics
        precision_score = tp / (tp + fp + 1e-8)
        recall_score = tp / (tp + fn + 1e-8)
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-8)
        
        # Business metrics
        false_positive_rate = fp / (fp + tn + 1e-8)
        
        # Precision@K and Recall@K (top 100 predictions)
        top_k = 100
        top_k_indices = np.argsort(y_pred_proba)[-top_k:]
        precision_at_k = y_true.iloc[top_k_indices].sum() / top_k
        recall_at_k = y_true.iloc[top_k_indices].sum() / y_true.sum()
        
        results = {
            'auc_score': auc_score,
            'pr_auc_score': pr_auc_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'precision_at_100': precision_at_k,
            'recall_at_100': recall_at_k,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        logger.info(f"{model_name} - AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}, "
                   f"F1: {f1_score:.4f}, Precision@100: {precision_at_k:.4f}")
        
        return results
    
    def _save_model(self, model: Any, model_name: str):
        """Save trained model."""
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} to {model_path}")
    
    def load_model(self, model_name: str) -> Any:
        """Load trained model."""
        model_path = self.models_dir / f"{model_name}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Loaded {model_name} from {model_path}")
            return model
        else:
            logger.warning(f"Model {model_name} not found at {model_path}")
            return None
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model based on PR-AUC."""
        if not self.results:
            raise ValueError("No models trained yet!")
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['pr_auc_score'])
        best_model = self.load_model(best_model_name)
        
        logger.info(f"Best model: {best_model_name} (PR-AUC: {self.results[best_model_name]['pr_auc_score']:.4f})")
        return best_model_name, best_model
    
    def generate_model_report(self) -> str:
        """Generate comprehensive model evaluation report."""
        if not self.results:
            return "No models trained yet!"
        
        report = "=" * 80 + "\n"
        report += "CREDIT CARD FRAUD DETECTION - MODEL EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Summary table
        report += "MODEL PERFORMANCE SUMMARY\n"
        report += "-" * 50 + "\n"
        report += f"{'Model':<20} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1':<10} {'Precision@100':<15}\n"
        report += "-" * 50 + "\n"
        
        for model_name, results in self.results.items():
            report += f"{model_name:<20} {results['pr_auc_score']:<10.4f} "
            report += f"{results['auc_score']:<10.4f} {results['f1_score']:<10.4f} "
            report += f"{results['precision_at_100']:<15.4f}\n"
        
        report += "\n"
        
        # Detailed results for each model
        for model_name, results in self.results.items():
            report += f"DETAILED RESULTS - {model_name.upper()}\n"
            report += "-" * 40 + "\n"
            report += f"Precision-Recall AUC: {results['pr_auc_score']:.4f}\n"
            report += f"ROC AUC: {results['auc_score']:.4f}\n"
            report += f"Precision: {results['precision']:.4f}\n"
            report += f"Recall: {results['recall']:.4f}\n"
            report += f"F1 Score: {results['f1_score']:.4f}\n"
            report += f"False Positive Rate: {results['false_positive_rate']:.4f}\n"
            report += f"Precision@100: {results['precision_at_100']:.4f}\n"
            report += f"Recall@100: {results['recall_at_100']:.4f}\n"
            report += f"Optimal Threshold: {results['optimal_threshold']:.4f}\n"
            
            cm = results['confusion_matrix']
            report += f"Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}\n"
            report += "\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'Time': np.random.exponential(1000, n_samples),
        'Amount': np.random.lognormal(3, 1.5, n_samples),
        'Class': np.random.binomial(1, 0.01, n_samples)
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Generate report
    report = trainer.generate_model_report()
    print(report)
