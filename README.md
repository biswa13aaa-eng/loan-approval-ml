# Loan Approval ML

An end-to-end Machine Learning pipeline for predicting loan approval decisions based on applicant financial and demographic profiles.

## Project Structure

```text
Loan-Approval-ML/
├── data/
│   ├── raw/                # Original benchmark dataset (loan_data.csv)
│   └── processed/          # Processed data outputs
├── models/
│   ├── best_loan_model.joblib  # Serialized best model pipeline
│   └── model_metadata.json     # Performance metrics and run details
├── notebooks/
│   └── 01_loan_eda.ipynb   # Comprehensive Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths, feature groups, and hyperparameter configs
│   ├── data_loader.py      # Data loading and stratified splitting
│   ├── preprocessing.py    # Custom feature engineering & ColumnTransformer pipeline
│   ├── evaluate.py         # Evaluation metrics and reporting
│   ├── train.py            # 5-fold CV across candidate models & artifact persistence
│   └── predict.py          # Real-time applicant inference & risk assessment
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup Instructions

1. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### 1. Run Model Training & Cross-Validation
Trains candidate models (`Logistic Regression`, `Random Forest`, `XGBoost`, `LightGBM`) with 5-fold stratified cross-validation, selects the best architecture, evaluates on the holdout test set, and saves model artifacts:
```powershell
python -m src.train
```

### 2. Run Real-Time Inference
Run predictions for applicant profiles with approval decisions, probabilities, and risk tiers:
```powershell
python -m src.predict
```

### 3. Exploratory Data Analysis (EDA)
Open and run [notebooks/01_loan_eda.ipynb](notebooks/01_loan_eda.ipynb) in VS Code or JupyterLab:
```powershell
jupyter lab
```

## Model Performance (Holdout Test Set)

| Metric | Random Forest (Best Model) |
| :--- | :--- |
| **Accuracy** | **83.74%** |
| **ROC-AUC** | **87.55%** |
| **Precision** | **89.16%** |
| **Recall** | **87.06%** |
| **F1-Score** | **88.10%** |
