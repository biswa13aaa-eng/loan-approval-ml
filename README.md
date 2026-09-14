# 🏦 Loan Approval Prediction & Credit Risk Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-11557C)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/Academic-Coursework-green)](#)

An end-to-end Machine Learning classification system designed to predict whether commercial loan applications are approved or rejected. The project includes data preprocessing, feature engineering, exploratory data analysis, multiple machine learning models, hyperparameter tuning, model evaluation, feature importance analysis, and an interactive Streamlit decision dashboard.

## 🚀 Live Demo

👉 **[Launch LoanWise AI – Live Streamlit App](https://loan-approval-ml-9jv7lfeebayttgckafqwzu.streamlit.app/)**

The live application allows users to explore the dataset, analyze model performance, inspect feature importance, and test loan approval predictions using applicant information.

---

## 1. Project Overview

In commercial retail lending, automated credit decisioning can help financial institutions process loan applications efficiently while assessing borrower risk.

This project develops and benchmarks supervised machine learning models to assess loan applications using applicant demographic information, employment details, requested loan information, and credit history.

---

## 2. Problem Statement

Manual loan underwriting can be time-intensive and difficult to scale. An automated machine learning system can assist in identifying patterns associated with loan approval and rejection.

- **Type of Machine Learning:** Supervised Learning
- **Problem Formulation:** Binary Classification
- **Target Variable:** `Loan_Status`
- **Target Classes:** `0 = Rejected (N)`, `1 = Approved (Y)`
- **Primary Evaluation Focus:** ROC-AUC, Precision, Recall, F1-Score, and Accuracy

---

## 3. Dataset

The project uses the **Loan Approval Classification Benchmark** dataset stored in:

`data/raw/loan_data.csv`

- **Total Records:** 614 applicant rows
- **Total Features:** 12 input features + 1 target variable
- **Duplicate Rows:** 0

### Class Distribution

- **Approved (`Y`):** 422 applicants (68.73%)
- **Rejected (`N`):** 192 applicants (31.27%)

### Attributes

- `Loan_ID` — Unique applicant identifier
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status` — Target variable

---

## 4. Technologies Used

| Technology | Version / Purpose |
|---|---|
| Python | 3.14 |
| Pandas | 3.0.5 |
| NumPy | 2.5.3 |
| Scikit-Learn | 1.9.1 |
| XGBoost | 3.4.1 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Plotly | 7.0.0 |
| Streamlit | 1.63.0 |
| ReportLab | 5.0.1 |
| Jupyter Notebook | Analysis & documentation |

---

## 5. Project Structure

```text
Loan-Approval-ML/
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_loan_eda.ipynb
│   └── loan_approval_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── build_notebook.py
│   └── generate_pdf_report.py
│
├── models/
│   ├── best_loan_model.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── model_metadata.json
│
├── reports/
│   ├── figures/
│   ├── results/
│   ├── loan_approval_report.pdf
│   └── viva_preparation.md
│
├── app/
│   ├── app.py
│   ├── components/
│   └── pages/
│
├── requirements.txt
├── README.md
└── .gitignore
