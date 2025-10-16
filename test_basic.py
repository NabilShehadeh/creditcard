#!/usr/bin/env python3
"""
Simple test script to verify basic functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from data.processor import DataProcessor
        print("✓ DataProcessor imported successfully")
        
        from features.engineering import FeatureEngineer
        print("✓ FeatureEngineer imported successfully")
        
        from models.trainer import ModelTrainer
        print("✓ ModelTrainer imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_processing():
    """Test basic data processing functionality."""
    try:
        from data.processor import DataProcessor
        
        processor = DataProcessor()
        df = processor.load_data()
        
        print(f"✓ Data loaded successfully: {df.shape}")
        print(f"✓ Fraud rate: {df['Class'].mean():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return False

def test_feature_engineering():
    """Test basic feature engineering functionality."""
    try:
        from features.engineering import FeatureEngineer
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        np.random.seed(42)
        data = {
            'Time': np.random.exponential(1000, 100),
            'Amount': np.random.lognormal(3, 1.5, 100),
            'Class': np.random.binomial(1, 0.01, 100)
        }
        
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, 100)
        
        df = pd.DataFrame(data)
        
        engineer = FeatureEngineer()
        df_features = engineer.engineer_all_features(df)
        
        print(f"✓ Feature engineering successful: {len(df_features.columns)} features")
        
        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running basic functionality tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_processing,
        test_feature_engineering
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
