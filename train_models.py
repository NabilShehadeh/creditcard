#!/usr/bin/env python3
"""
Training script for credit card fraud detection models.

This script demonstrates how to train all models and evaluate their performance.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.processor import DataProcessor
from models.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    logger.info("Starting credit card fraud detection model training...")
    
    try:
        # Initialize data processor
        processor = DataProcessor()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = processor.load_data()
        df_processed = processor.preprocess_data(df)
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        train_df, test_df = processor.split_data(df_processed)
        
        # Prepare features and targets
        logger.info("Preparing features and targets...")
        X_train, y_train = processor.prepare_features_target(train_df)
        X_val, y_val = processor.prepare_features_target(test_df)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Training fraud rate: {y_train.mean():.4f}")
        logger.info(f"Validation fraud rate: {y_val.mean():.4f}")
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Train all models
        logger.info("Training models...")
        results = trainer.train_all_models(X_train, y_train, X_val, y_val)
        
        # Generate and display report
        logger.info("Generating model evaluation report...")
        report = trainer.generate_model_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model()
        logger.info(f"Best model: {best_model_name}")
        
        # Save preprocessor
        processor.save_preprocessor("models/preprocessor.joblib")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
