# Anomaly Detection Platform

A unified machine learning platform for financial fraud detection and network intrusion detection. It includes model training pipelines, a FastAPI backend, and an interactive Streamlit dashboard.

## Project Overview

- Fraud detection using the IEEE-CIS fraud dataset
- Network intrusion detection using NSL-KDD
- XGBoost as the primary supervised model
- Isolation Forest as an anomaly scoring layer for fraud
- SHAP explainability utilities
- REST API for predictions
- Streamlit dashboard for single-record and CSV prediction workflows

## Model Performance

### Fraud Detection (IEEE-CIS Dataset)

These fraud metrics are from a 200,000-row IEEE-CIS training run.

| Model | ROC-AUC | Avg Precision |
| --- | ---: | ---: |
| Logistic Regression | 0.8802 | 0.4326 |
| XGBoost | 0.9411 | 0.6796 |
| XGBoost + Isolation Forest | 0.9373 | - |

The IEEE-CIS fraud pipeline handles missing values, categorical columns, numeric scaling, and class imbalance. XGBoost uses `scale_pos_weight` instead of expanding the large transformed feature matrix with SMOTE.

### Intrusion Detection (NSL-KDD Dataset)

| Model | ROC-AUC | Avg Precision |
| --- | ---: | ---: |
| Random Forest | 0.9691 | 0.9699 |
| XGBoost | 0.9696 | 0.9725 |

## Repository Structure

```text
anomaly-detection-platform/
├── data/
│   └── raw/
│       ├── fraud/      # IEEE-CIS CSV files
│       └── intrusion/  # NSL-KDD Train/Test files
├── models/
│   ├── fraud/
│   └── intrusion/
├── reports/
│   └── figures/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── fraud/
│   ├── intrusion/
│   └── shared/
└── tests/
```

## Setup

Using `D:\anomaly-detection-platform` is recommended for large datasets and model artifacts.

```powershell
cd D:\anomaly-detection-platform
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

Place the IEEE-CIS fraud files in `data/raw/fraud/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Place the NSL-KDD files in `data/raw/intrusion/`:

- `KDDTrain+.txt`
- `KDDTest+.txt`

Raw datasets are git-ignored because they are large.

## Training

For practical local training, start with a sample:

```powershell
$env:FRAUD_SAMPLE_ROWS=200000
python src/fraud/train.py
python src/fraud/anomaly.py
```

To train on the full IEEE-CIS training file:

```powershell
Remove-Item Env:FRAUD_SAMPLE_ROWS
python src/fraud/train.py
python src/fraud/anomaly.py
```

Optional fraud explainability plots:

```powershell
python src/fraud/explain.py
```

Intrusion training:

```powershell
python src/intrusion/train.py
python src/intrusion/explain.py
```

## Running The App

Start the API in one terminal:

```powershell
cd D:\anomaly-detection-platform
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --port 8000
```

Start the Streamlit dashboard in another terminal:

```powershell
cd D:\anomaly-detection-platform
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
streamlit run src/dashboard/app.py
```

Open:

```text
http://localhost:8501
```

The sidebar should show `API Connected`.

## API Reference

### Health Check

```http
GET /health
```

### Fraud Prediction

```http
POST /predict/fraud
Content-Type: application/json

{
  "record": {
    "TransactionAmt": 68.5,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "discover",
    "card6": "credit"
  }
}
```

Response:

```json
{
  "prediction": 0,
  "label": "NORMAL",
  "fraud_probability": 0.1874,
  "anomaly_score": 0.3153,
  "combined_risk_score": 0.2257,
  "risk_level": "MINIMAL"
}
```

Batch fraud prediction is supported with:

```json
{
  "records": [
    {
      "TransactionAmt": 68.5,
      "ProductCD": "W",
      "card1": 13926
    }
  ]
}
```

### Intrusion Prediction

```http
POST /predict/intrusion
Content-Type: application/json

{
  "features": [0.0, 1.0, 0.0]
}
```

The intrusion endpoint expects all 41 NSL-KDD feature values.

## Dashboard Notes

The fraud dashboard now uses IEEE-CIS fields such as:

- `TransactionAmt`
- `ProductCD`
- `card1`
- `card2`
- `card4`
- `card6`
- `C1`
- `C2`
- `D1`
- `DeviceType`

For CSV batch prediction, use a small sample CSV first. The current dashboard processes rows sequentially, so uploading full IEEE files can take a long time.

## Key Implementation Details

- `src/fraud/preprocessor.py` loads IEEE transaction and identity files, joins them on `TransactionID`, and builds the preprocessing pipeline.
- `models/fraud/fraud_preprocessor.pkl` is required at inference time.
- `src/api/main.py` accepts raw IEEE-CIS fraud records through `record` or `records`.
- `src/dashboard/app.py` uses IEEE-CIS sample fields for the fraud UI.

## Tech Stack

| Layer | Technology |
| --- | --- |
| ML Models | XGBoost, Scikit-learn, Isolation Forest |
| Explainability | SHAP |
| Imbalance Handling | XGBoost `scale_pos_weight` |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Model Persistence | Joblib |

## Author

Preeti  
B.Tech Information Technology, NIT Kurukshetra  
[GitHub](https://github.com/silent-knight-22)
