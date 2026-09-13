# Loan Approval ML

An end-to-end Machine Learning project for predicting loan approval decisions based on applicant financial and demographic attributes.

## Project Structure

```text
Loan-Approval-ML/
├── data/
│   ├── raw/            # Untouched original datasets
│   └── processed/      # Cleaned and feature-engineered datasets
├── models/             # Serialized trained models and encoders
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/                # Modular application source code
│   ├── __init__.py
│   └── config.py       # Global project path configurations
├── .gitignore          # Git ignore rules for ML environments and artifacts
├── README.md           # Project overview and documentation
└── requirements.txt    # Python dependencies
```

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
