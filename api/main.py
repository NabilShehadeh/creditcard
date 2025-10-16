"""
FastAPI service for credit card fraud detection.

This module provides a REST API for real-time fraud detection with
explainability features and monitoring capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import joblib
from pathlib import Path
import asyncio
from datetime import datetime
import json

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Model explainability
import shap

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.trainer import ModelTrainer
from data.processor import DataProcessor

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection with explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and explainer
model = None
explainer = None
preprocessor = None
model_name = None

# Pydantic models for API
class TransactionRequest(BaseModel):
    """Transaction data for fraud prediction."""
    
    # Core transaction features
    Time: float = Field(..., description="Time in seconds since first transaction")
    Amount: float = Field(..., description="Transaction amount", gt=0)
    
    # V-features (anonymized PCA components)
    V1: float = Field(..., description="Anonymized feature V1")
    V2: float = Field(..., description="Anonymized feature V2")
    V3: float = Field(..., description="Anonymized feature V3")
    V4: float = Field(..., description="Anonymized feature V4")
    V5: float = Field(..., description="Anonymized feature V5")
    V6: float = Field(..., description="Anonymized feature V6")
    V7: float = Field(..., description="Anonymized feature V7")
    V8: float = Field(..., description="Anonymized feature V8")
    V9: float = Field(..., description="Anonymized feature V9")
    V10: float = Field(..., description="Anonymized feature V10")
    V11: float = Field(..., description="Anonymized feature V11")
    V12: float = Field(..., description="Anonymized feature V12")
    V13: float = Field(..., description="Anonymized feature V13")
    V14: float = Field(..., description="Anonymized feature V14")
    V15: float = Field(..., description="Anonymized feature V15")
    V16: float = Field(..., description="Anonymized feature V16")
    V17: float = Field(..., description="Anonymized feature V17")
    V18: float = Field(..., description="Anonymized feature V18")
    V19: float = Field(..., description="Anonymized feature V19")
    V20: float = Field(..., description="Anonymized feature V20")
    V21: float = Field(..., description="Anonymized feature V21")
    V22: float = Field(..., description="Anonymized feature V22")
    V23: float = Field(..., description="Anonymized feature V23")
    V24: float = Field(..., description="Anonymized feature V24")
    V25: float = Field(..., description="Anonymized feature V25")
    V26: float = Field(..., description="Anonymized feature V26")
    V27: float = Field(..., description="Anonymized feature V27")
    V28: float = Field(..., description="Anonymized feature V28")
    
    # Optional metadata
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    card_id: Optional[str] = Field(None, description="Card/account identifier")
    
    @validator('Amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class PredictionResponse(BaseModel):
    """Response model for fraud predictions."""
    
    transaction_id: Optional[str]
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    fraud_prediction: bool = Field(..., description="Binary fraud prediction")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    confidence: float = Field(..., description="Model confidence score")
    
    # Explainability
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    feature_importance: Dict[str, float] = Field(..., description="All feature importance scores")
    
    # Metadata
    model_name: str = Field(..., description="Name of the model used")
    prediction_time: str = Field(..., description="Timestamp of prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    transactions: List[TransactionRequest] = Field(..., description="List of transactions")
    return_explanations: bool = Field(True, description="Whether to return feature explanations")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    processing_time_ms: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    timestamp: str = Field(..., description="Current timestamp")


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the model and explainer on startup."""
    global model, explainer, preprocessor, model_name
    
    try:
        logger.info("Starting fraud detection API...")
        
        # Load the best model
        trainer = ModelTrainer()
        model_name, model = trainer.get_best_model()
        
        if model is None:
            logger.warning("No trained model found. Training a new model...")
            await train_and_load_model()
        else:
            logger.info(f"Loaded model: {model_name}")
        
        # Initialize explainer
        await initialize_explainer()
        
        # Load preprocessor
        preprocessor = DataProcessor()
        
        logger.info("API startup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


async def train_and_load_model():
    """Train a new model if none exists."""
    global model, model_name
    
    try:
        logger.info("Training new model...")
        
        # Generate sample data for training
        processor = DataProcessor()
        df = processor.load_data()
        df_processed = processor.preprocess_data(df)
        
        # Split data
        train_df, test_df = processor.split_data(df_processed)
        
        # Prepare features
        X_train, y_train = processor.prepare_features_target(train_df)
        X_val, y_val = processor.prepare_features_target(test_df)
        
        # Train models
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, y_train, X_val, y_val)
        
        # Get best model
        model_name, model = trainer.get_best_model()
        
        logger.info(f"Model training completed. Best model: {model_name}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


async def initialize_explainer():
    """Initialize SHAP explainer for the model."""
    global explainer
    
    try:
        if model is None:
            logger.warning("No model loaded, cannot initialize explainer")
            return
        
        # Generate background data for SHAP
        processor = DataProcessor()
        df = processor.load_data()
        df_processed = processor.preprocess_data(df)
        X, _ = processor.prepare_features_target(df_processed)
        
        # Sample background data
        background_data = X.sample(min(100, len(X)), random_state=42)
        
        # Initialize explainer based on model type
        if model_name == 'lightgbm':
            explainer = shap.TreeExplainer(model)
        elif model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            explainer = shap.Explainer(model, background_data)
        else:
            logger.warning(f"SHAP explainer not supported for {model_name}")
            explainer = None
        
        logger.info(f"SHAP explainer initialized for {model_name}")
        
    except Exception as e:
        logger.error(f"Error initializing explainer: {str(e)}")
        explainer = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Credit Card Fraud Detection API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_name,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest, background_tasks: BackgroundTasks):
    """Predict fraud for a single transaction."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert transaction to DataFrame
        transaction_data = transaction.dict()
        transaction_id = transaction_data.pop('transaction_id', None)
        card_id = transaction_data.pop('card_id', None)
        
        df = pd.DataFrame([transaction_data])
        
        # Preprocess transaction
        df_processed = preprocessor.preprocess_data(df)
        X, _ = preprocessor.prepare_features_target(df_processed)
        
        # Make prediction
        if model_name == 'lightgbm':
            fraud_probability = model.predict(X)[0]
        elif model_name == 'autoencoder':
            # Special handling for autoencoder
            scaler = model['scaler']
            X_scaled = scaler.transform(X)
            X_reconstructed = model['model'].predict(X_scaled)
            reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            fraud_probability = reconstruction_error[0] / np.max(reconstruction_error)
        else:
            fraud_probability = model.predict_proba(X)[0, 1]
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "LOW"
        elif fraud_probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Binary prediction
        fraud_prediction = fraud_probability > 0.5
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(fraud_probability - 0.5) * 2
        
        # Generate explanations
        top_features, feature_importance = await generate_explanations(X)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            transaction_id,
            fraud_probability,
            fraud_prediction,
            risk_level
        )
        
        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=float(fraud_probability),
            fraud_prediction=bool(fraud_prediction),
            risk_level=risk_level,
            confidence=float(confidence),
            top_features=top_features,
            feature_importance=feature_importance,
            model_name=model_name,
            prediction_time=start_time.isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    predictions = []
    
    try:
        for transaction in request.transactions:
            # Convert to single prediction
            single_response = await predict_fraud(transaction)
            predictions.append(single_response)
        
        # Calculate summary statistics
        fraud_probabilities = [p.fraud_probability for p in predictions]
        fraud_predictions = [p.fraud_prediction for p in predictions]
        
        summary = {
            "total_transactions": len(predictions),
            "fraud_predictions": sum(fraud_predictions),
            "fraud_rate": sum(fraud_predictions) / len(predictions),
            "avg_fraud_probability": np.mean(fraud_probabilities),
            "max_fraud_probability": np.max(fraud_probabilities),
            "min_fraud_probability": np.min(fraud_probabilities)
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


async def generate_explanations(X: pd.DataFrame) -> tuple:
    """Generate feature explanations using SHAP."""
    
    if explainer is None:
        # Fallback: use correlation-based importance
        feature_importance = {}
        for col in X.columns:
            feature_importance[col] = abs(X[col].iloc[0])  # Simple magnitude-based importance
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [
            {"feature": feat, "value": val, "importance": val}
            for feat, val in sorted_features[:5]
        ]
        
        return top_features, feature_importance
    
    try:
        # Generate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class values
        
        # Calculate feature importance
        feature_importance = {}
        for i, col in enumerate(X.columns):
            feature_importance[col] = abs(shap_values[0][i])
        
        # Get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [
            {
                "feature": feat,
                "value": float(X[feat].iloc[0]),
                "importance": float(val),
                "shap_value": float(shap_values[0][X.columns.get_loc(feat)])
            }
            for feat, val in sorted_features[:5]
        ]
        
        return top_features, feature_importance
        
    except Exception as e:
        logger.warning(f"Error generating SHAP explanations: {str(e)}")
        
        # Fallback to simple importance
        feature_importance = {}
        for col in X.columns:
            feature_importance[col] = abs(X[col].iloc[0])
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [
            {"feature": feat, "value": val, "importance": val}
            for feat, val in sorted_features[:5]
        ]
        
        return top_features, feature_importance


async def log_prediction(transaction_id: str, fraud_probability: float, 
                        fraud_prediction: bool, risk_level: str):
    """Log prediction for monitoring purposes."""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "fraud_prediction": fraud_prediction,
        "risk_level": risk_level,
        "model_name": model_name
    }
    
    logger.info(f"Prediction logged: {json.dumps(log_entry)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features": list(preprocessor.feature_columns) if preprocessor.feature_columns else [],
        "explainer_available": explainer is not None,
        "last_updated": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # This would typically come from a metrics store
    # For now, return placeholder metrics
    return {
        "model_name": model_name,
        "total_predictions": 0,  # Would be tracked in production
        "fraud_predictions": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "last_updated": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
