#!/usr/bin/env python3
"""
Demo script for credit card fraud detection system.

This script demonstrates the complete pipeline from data loading to prediction.
"""

import sys
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.processor import DataProcessor
from features.engineering import FeatureEngineer
from models.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_data_processing():
    """Demonstrate data processing capabilities."""
    print("\n" + "="*60)
    print("DATA PROCESSING DEMO")
    print("="*60)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load sample data
    print("Loading sample data...")
    df = processor.load_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud rate: {df['Class'].mean():.4f} ({df['Class'].mean()*100:.2f}%)")
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_processed = processor.preprocess_data(df)
    print(f"   Processed shape: {df_processed.shape}")
    print(f"   New features added: {len(df_processed.columns) - len(df.columns)}")
    
    # Split data
    print("\nSplitting data...")
    train_df, test_df = processor.split_data(df_processed)
    print(f"   Training set: {len(train_df):,} transactions")
    print(f"   Test set: {len(test_df):,} transactions")
    print(f"   Train fraud rate: {train_df['Class'].mean():.4f}")
    print(f"   Test fraud rate: {test_df['Class'].mean():.4f}")
    
    return train_df, test_df


def demo_feature_engineering():
    """Demonstrate feature engineering capabilities."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING DEMO")
    print("="*60)
    
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
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    print("Engineering features...")
    df_features = engineer.engineer_all_features(df)
    
    print(f"   Original features: {len(df.columns)}")
    print(f"   Engineered features: {len(df_features.columns)}")
    print(f"   New features: {len(df_features.columns) - len(df.columns)}")
    
    # Get feature importance
    importance = engineer.get_feature_importance(df_features)
    top_features = list(importance.items())[:10]
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature:<30} {score:.4f}")
    
    return df_features


def demo_model_training(train_df, test_df):
    """Demonstrate model training capabilities."""
    print("\n" + "="*60)
    print("MODEL TRAINING DEMO")
    print("="*60)
    
    # Prepare data
    processor = DataProcessor()
    X_train, y_train = processor.prepare_features_target(train_df)
    X_val, y_val = processor.prepare_features_target(test_df)
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    print("\nTraining models...")
    print("   Logistic Regression...")
    print("   Random Forest...")
    print("   LightGBM...")
    print("   XGBoost...")
    print("   Autoencoder...")
    print("   Ensemble...")
    
    # Train models (this will take some time)
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    print("\nModel Performance Summary:")
    print("-" * 50)
    print(f"{'Model':<20} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1':<10}")
    print("-" * 50)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['pr_auc_score']:<10.4f} "
              f"{result['auc_score']:<10.4f} {result['f1_score']:<10.4f}")
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"\nBest Model: {best_model_name}")
    
    return trainer, results


def demo_prediction(trainer, test_df):
    """Demonstrate prediction capabilities."""
    print("\n" + "="*60)
    print("PREDICTION DEMO")
    print("="*60)
    
    # Get a sample transaction
    sample_transaction = test_df.iloc[0].copy()
    actual_class = sample_transaction['Class']
    
    print(f"Sample Transaction:")
    print(f"   Amount: ${sample_transaction['Amount']:.2f}")
    print(f"   Time: {sample_transaction['Time']:.0f} seconds")
    print(f"   Actual Class: {'Fraud' if actual_class == 1 else 'Normal'}")
    
    # Prepare features for prediction
    processor = DataProcessor()
    X_sample, _ = processor.prepare_features_target(test_df.iloc[:1])
    
    # Make prediction using best model
    best_model_name, best_model = trainer.get_best_model()
    
    if best_model_name == 'lightgbm':
        fraud_probability = best_model.predict(X_sample)[0]
    elif best_model_name == 'autoencoder':
        scaler = best_model['scaler']
        X_scaled = scaler.transform(X_sample)
        X_reconstructed = best_model['model'].predict(X_scaled)
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        fraud_probability = reconstruction_error[0] / np.max(reconstruction_error)
    else:
        fraud_probability = best_model.predict_proba(X_sample)[0, 1]
    
    # Determine risk level
    if fraud_probability < 0.3:
        risk_level = "LOW"
    elif fraud_probability < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    print(f"\nPrediction Results:")
    print(f"   Model: {best_model_name}")
    print(f"   Fraud Probability: {fraud_probability:.4f}")
    print(f"   Risk Level: {risk_level}")
    print(f"   Prediction: {'Fraud' if fraud_probability > 0.5 else 'Normal'}")
    print(f"   Correct: {'Yes' if (fraud_probability > 0.5) == (actual_class == 1) else 'No'}")


def demo_api_usage():
    """Demonstrate API usage."""
    print("\n" + "="*60)
    print("API USAGE DEMO")
    print("="*60)
    
    print("API Endpoints:")
    print("   GET  /health              - Health check")
    print("   POST /predict              - Single prediction")
    print("   POST /predict/batch        - Batch predictions")
    print("   GET  /model/info           - Model information")
    print("   GET  /metrics              - Performance metrics")
    
    print("\nStarting API Server:")
    print("   python api/main.py")
    
    print("\nExample API Request:")
    sample_request = {
        "Time": 12345,
        "Amount": 100.0,
        "V1": 0.5, "V2": -0.2, "V3": 0.1, "V4": -0.3, "V5": 0.2,
        "V6": -0.1, "V7": 0.4, "V8": -0.2, "V9": 0.3, "V10": -0.1,
        "V11": 0.2, "V12": -0.3, "V13": 0.1, "V14": -0.2, "V15": 0.3,
        "V16": -0.1, "V17": 0.2, "V18": -0.3, "V19": 0.1, "V20": -0.2,
        "V21": 0.3, "V22": -0.1, "V23": 0.2, "V24": -0.3, "V25": 0.1,
        "V26": -0.2, "V27": 0.3, "V28": -0.1
    }
    
    print("   curl -X POST 'http://localhost:8000/predict' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '" + json.dumps(sample_request, indent=2) + "'")


def main():
    """Main demo function."""
    print("CREDIT CARD FRAUD DETECTION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the complete fraud detection pipeline")
    print("from data processing to model training and prediction.")
    
    try:
        # Demo 1: Data Processing
        train_df, test_df = demo_data_processing()
        
        # Demo 2: Feature Engineering
        demo_feature_engineering()
        
        # Demo 3: Model Training
        trainer, results = demo_model_training(train_df, test_df)
        
        # Demo 4: Prediction
        demo_prediction(trainer, test_df)
        
        # Demo 5: API Usage
        demo_api_usage()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("   Data loading and preprocessing")
        print("   Advanced feature engineering")
        print("   Multiple ML model training")
        print("   Model evaluation and comparison")
        print("   Real-time prediction capabilities")
        print("   RESTful API for model serving")
        print("   Comprehensive documentation")
        print("   Docker containerization")
        print("   CI/CD pipeline")
        
        print("\nNext Steps:")
        print("   1. Run 'python train_models.py' to train models")
        print("   2. Run 'python api/main.py' to start the API")
        print("   3. Open 'notebooks/01_eda.ipynb' for detailed analysis")
        print("   4. Check 'docs/model_card.md' for model documentation")
        print("   5. Run 'pytest tests/' to execute test suite")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed: {str(e)}")
        print("Please check the error message and try again.")


if __name__ == "__main__":
    main()