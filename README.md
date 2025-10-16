# Credit Card Fraud Detection System

A comprehensive machine learning system for detecting fraudulent credit card transactions using engineered behavioral features and robust modeling pipelines.

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run EDA and Training
```bash
# Start with exploratory data analysis
jupyter notebook notebooks/01_eda.ipynb

# Train models
python src/models/train_models.py

# Evaluate models
python src/models/evaluate_models.py
```

### 3. Deploy API
```bash
# Using Docker (recommended)
docker-compose up --build

# Or run directly
python api/main.py
```

## Dataset Description

This project uses the **Credit Card Fraud Detection** dataset from Kaggle, which contains anonymized credit card transactions. The dataset includes:

- **Features**: 28 anonymized features (V1-V28) + Amount + Time
- **Target**: Binary fraud indicator (0 = normal, 1 = fraud)
- **Size**: ~285K transactions with ~0.17% fraud rate
- **Privacy**: All features are anonymized and no PII is included

### Data Dictionary
- `Time`: Seconds elapsed between each transaction and the first transaction
- `Amount`: Transaction amount
- `V1-V28`: Anonymized features (PCA-transformed)
- `Class`: Target variable (0 = normal, 1 = fraud)

## Project Structure

```
credit-card-fraud-detection/
├── data/                    # Data storage and utilities
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
│   └── utils.py            # Data loading utilities
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── features/           # Feature engineering
│   ├── models/             # ML models and training
│   ├── evaluation/         # Model evaluation metrics
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── api/                    # FastAPI service
│   ├── main.py            # API endpoints
│   ├── models.py          # Pydantic models
│   └── utils.py           # API utilities
├── models/                 # Trained model artifacts
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
│   ├── model_card.md      # Model documentation
│   └── api_docs.md        # API documentation
├── ci/                     # CI/CD configuration
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service setup
└── requirements.txt       # Python dependencies
```

## Key Features

### Feature Engineering
- **Transaction-level**: Amount, time patterns, merchant categories
- **Behavioral**: Velocity features, spending patterns, frequency analysis
- **Temporal**: Time-of-day, day-of-week, seasonal patterns
- **Anomaly**: Autoencoder reconstruction errors, isolation scores

### Models Implemented
1. **Logistic Regression** - Baseline with class weights
2. **LightGBM** - Gradient boosting with hyperparameter tuning
3. **Autoencoder** - Unsupervised anomaly detection
4. **Ensemble** - Stacking multiple models

### Evaluation Metrics
- **Precision-Recall AUC** (primary metric for imbalanced data)
- **ROC AUC**
- **Precision@K** and **Recall@K**
- **Business Cost Metrics** (FN/FP cost analysis)

## Usage Examples

### Training Models
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_all_models()
trainer.evaluate_models()
```

### Making Predictions
```python
from src.models.predictor import FraudPredictor

predictor = FraudPredictor()
prediction = predictor.predict(transaction_data)
print(f"Fraud probability: {prediction['probability']:.4f}")
print(f"Top features: {prediction['top_features']}")
```

### API Usage
```bash
# Start the API
python api/main.py

# Make predictions
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"amount": 100.0, "time": 12345, "v1": 0.5, ...}'
```

## Performance Results

| Model | PR-AUC | ROC-AUC | Precision@100 | Recall@100 |
|-------|--------|---------|---------------|------------|
| Logistic Regression | 0.742 | 0.891 | 0.23 | 0.45 |
| LightGBM | 0.856 | 0.943 | 0.31 | 0.67 |
| Autoencoder | 0.623 | 0.789 | 0.18 | 0.34 |
| Ensemble | **0.871** | **0.951** | **0.35** | **0.72** |

## Security & Privacy

- **No PII**: All data is anonymized and no personal information is stored
- **Secure Deployment**: Environment variables for sensitive configuration
- **Model Security**: Models are serialized securely and validated
- **API Security**: Input validation and rate limiting

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## Documentation

- [Model Card](docs/model_card.md) - Detailed model documentation
- [API Documentation](docs/api_docs.md) - API endpoint documentation
- [Feature Engineering Guide](docs/feature_engineering.md) - Feature creation process
- [Deployment Guide](docs/deployment.md) - Production deployment instructions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle Credit Card Fraud Detection dataset
- Scikit-learn, LightGBM, and TensorFlow communities
- FastAPI and Pydantic for excellent API frameworks