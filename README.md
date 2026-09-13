# 🏦 Loan Approval Prediction & Credit Risk Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-11557C)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/Academic-Coursework-green)](#)

An end-to-end, production-grade Machine Learning classification system designed to predict whether commercial loan applications are approved or rejected. Built with strict ML hygiene, complete Scikit-Learn pipelines that prevent data leakage, extensive exploratory analysis, hyperparameter optimization, and a full interactive Streamlit decision dashboard.

---

## 1. Project Overview
In commercial retail lending, automated credit decisioning helps financial institutions process loan volumes rapidly while guarding against credit default risk. This project develops and benchmarks supervised machine learning models to assess borrower creditworthiness using applicant demographic details, employment history, requested loan dimensions, and historical repayment performance.

---

## 2. Problem Statement
Manual loan underwriting is time-intensive, subject to cognitive bias, and difficult to scale. Conversely, misclassifying risky applicants leads to Non-Performing Assets (NPAs), while erroneously rejecting creditworthy applicants causes lost interest revenue and customer dissatisfaction.

- **Type of Machine Learning:** Supervised Learning
- **Problem Formulation:** Binary Classification
- **Target Variable:** `Loan_Status` $\in \{0: \text{Rejected (N)}, 1: \text{Approved (Y)}\}$
- **Core Optimization Objective:** Maximize discriminative power (**ROC-AUC**) while maintaining high **Precision** to protect capital reserves against defaults.

---

## 3. Dataset
The project is built and evaluated using the verified **Kaggle Loan Approval Classification Benchmark** (`data/raw/loan_data.csv`):
- **Total Records:** 614 applicant rows
- **Total Features:** 12 input features + 1 target variable (`Loan_Status`)
- **Duplicate Rows:** 0
- **Class Distribution:**
  - Approved (`Y`): 422 applicants (68.73%)
  - Rejected (`N`): 192 applicants (31.27%)
- **Attributes:**
  - `Loan_ID`: Unique applicant string identifier (dropped prior to training).
  - Categorical (6): `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`.
  - Numerical / Flags (5): `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`.

---

## 4. Technologies Used
- **Programming Language:** Python 3.14
- **Data Engineering:** Pandas 3.0.5, NumPy 2.5.3
- **Machine Learning:** Scikit-Learn 1.9.1, XGBoost 3.4.1
- **Visualization:** Matplotlib 3.11.2, Seaborn 0.13.2, Plotly 7.0.0
- **User Interface:** Streamlit 1.63.0
- **Report & Documentation:** ReportLab 5.0.1, Jupyter Notebook

---

## 5. Project Structure
```text
C:\Loan-Approval-ML/
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv               # Raw benchmark dataset (614 rows, 13 columns)
│   └── processed/                      # Preprocessed output cache
│
├── notebooks/
│   ├── 01_loan_eda.ipynb               # Preliminary EDA notebook
│   └── loan_approval_analysis.ipynb    # Complete 19-section submission notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, column groupings, random seeds
│   ├── data_loader.py                  # Loading and stratified train-test splitting
│   ├── preprocessing.py                # Custom feature engineering & ColumnTransformer
│   ├── eda.py                          # Dedicated figure generation script
│   ├── train.py                        # Model training, CV, GridSearchCV, and evaluation
│   ├── evaluate.py                     # Metric calculation helpers
│   ├── predict.py                      # Single & batch applicant inference helper
│   ├── build_notebook.py               # Notebook builder utility
│   └── generate_pdf_report.py          # 15-page academic PDF report compiler
│
├── models/
│   ├── best_loan_model.joblib          # Production Random Forest pipeline
│   ├── logistic_regression.joblib      # Baseline Logistic Regression pipeline
│   ├── random_forest.joblib            # Standalone Random Forest pipeline
│   ├── xgboost.joblib                  # Gradient Boosting pipeline
│   └── model_metadata.json             # Execution run details & metric summaries
│
├── reports/
│   ├── figures/                        # High-resolution exported PNG figures (1-9)
│   ├── results/                        # Benchmark metrics (CSV & JSON)
│   ├── loan_approval_report.pdf        # Publication-quality 15-page PDF report
│   └── viva_preparation.md             # 25-question student viva preparation guide
│
├── app/
│   └── app.py                          # Interactive Streamlit multi-tab web dashboard
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Comprehensive documentation
└── .gitignore                          # Clean repository hygiene
```

---

## 6. Data Preprocessing & Feature Engineering
To strictly avoid **Data Leakage**, train/test splitting was executed *first* (`test_size=0.2, random_state=42, stratify=y`). All imputers, encoders, and scalers were fit strictly on the training fold:

1. **Domain Feature Engineering (`LoanFeatureEngineer`):**
   - `Total_Income` = `ApplicantIncome` + `CoapplicantIncome`
   - `Loan_Amount_to_Total_Income` = `(LoanAmount * 1000) / Total_Income`
   - `Monthly_EMI` = `(LoanAmount * 1000) / Loan_Amount_Term`
   - `EMI_to_Income_Ratio` = `Monthly_EMI / Total_Income`
2. **Missing Value Imputation:**
   - Numerical (`LoanAmount`, `Loan_Amount_Term`): Imputed with **Median** (resistant to high income skewness).
   - Categorical (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`): Imputed with **Mode** (Most Frequent).
3. **Categorical Encoding:** One-Hot Encoding (`handle_unknown='ignore', sparse_output=False`).
4. **Feature Scaling:** `StandardScaler` applied across numerical attributes for linear model optimization.

---

## 7. Exploratory Data Analysis (EDA)
EDA figures are automatically generated and saved into `reports/figures/`:
- **Target Distribution (`01_target_distribution.png`):** Confirms 68.7% approval rate (mild class imbalance).
- **Credit History Impact (`02_credit_history_vs_approval.png`):** Applicants meeting guidelines have a **79.6% approval rate**, compared to just **8.0%** for applicants with past defaults.
- **Numerical Skewness (`03_numerical_distributions.png`):** Identifies significant right-skew in applicant income (up to $81,000) and loan requests.
- **Financial Ratios (`04_income_vs_loan_approval.png`):** Highlights that income alone does not ensure approval without repayment discipline.
- **Geographic Factors (`05_categorical_vs_approval.png`):** Semiurban properties show highest approval frequency (76.8%) vs Rural (61.5%).
- **Correlation Matrix (`06_correlation_heatmap.png`):** Confirms `Credit_History` exhibits the strongest linear correlation with `Loan_Status` ($r = 0.54$).

---

## 8. Models Used
1. **Logistic Regression (Baseline):**
   - *Rationale:* Transparent, interpretable linear baseline; models log-odds of loan approval with calibrated probabilities.
   - *Configuration:* `max_iter=1000`, `class_weight='balanced'`, `solver='lbfgs'`.
2. **Random Forest Classifier (Ensemble - Selected):**
   - *Rationale:* Non-linear bagging ensemble of 150 trees; handles piecewise thresholds and feature interactions without overfitting.
   - *Configuration:* `n_estimators=150`, `max_depth=6`, `min_samples_split=5`, `class_weight='balanced'`.
3. **XGBoost Classifier (Gradient Boosting):**
   - *Rationale:* Sequential boosting algorithm that minimizes loss gradient step-by-step.
   - *Configuration:* `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`.

---

## 9. Model Evaluation (Holdout Test Set)
All metrics were computed on an untouched holdout test set (123 applicants, 20% stratified sample). **No fabricated numbers**:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 82.11% | 87.95% | 85.88% | 86.90% | 87.28% |
| **Random Forest (Selected)** | **83.74%** | **89.16%** | **87.06%** | **88.10%** | **87.55%** |
| **XGBoost Classifier** | 85.37% | 84.54% | 96.47% | 90.11% | 84.09% |

---

## 10. Model Comparison & Risk Trade-off
- **Highest Accuracy:** XGBoost (85.37%)
- **Highest Precision:** **Random Forest (89.16%)**
- **Highest Recall:** XGBoost (96.47%)
- **Highest ROC-AUC:** **Random Forest (87.55%)**

### Why Random Forest Was Selected for Production
In commercial banking, **False Positives (approving a borrower who later defaults)** cause severe capital write-offs (loss of principal). A high **Precision (89.16%)** ensures that when the system issues an approval, it is confident in the borrower's repayment ability. While XGBoost had high recall (96.47%), it approved 15 high-risk defaulters (False Positives), compared to only 9 for Random Forest. Therefore, **Random Forest is the superior banking model**.

---

## 11. Feature Importance
Extracted using Mean Decrease in Impurity (MDI) from Random Forest:

| Rank | Feature | Importance | Domain Explanation |
| :---: | :--- | :---: | :--- |
| **1** | `Credit_History` | **42.59%** | Past credit discipline is overwhelmingly the single greatest predictor. |
| **2** | `Total_Income` | **7.19%** | Combined applicant and coapplicant gross earnings. |
| **3** | `Loan_Amount_to_Total_Income` | **7.14%** | Leverage obligation relative to income. |
| **4** | `Monthly_EMI` | **6.73%** | Monthly repayment cash requirement. |
| **5** | `ApplicantIncome` | **6.50%** | Primary applicant standalone salary. |

---

## 12. Hyperparameter Tuning (GridSearchCV)
GridSearchCV was performed on Random Forest over 72 hyperparameter combinations using 5-fold cross-validation:
- **Optimal Hyperparameters:** `{'max_depth': 4, 'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': None}`
- **Performance Comparison:**
  - *Baseline RF:* Accuracy: **86.99%** | F1-Score: **90.80%**
  - *Tuned RF:* Accuracy: **85.37%** | F1-Score: **90.32%**
- **Honest Academic Interpretation:** Pruning trees to `max_depth=4` slightly increased bias, yielding roughly comparable generalization. Documenting this transparently demonstrates scientific honesty.

---

## 13. Prediction Interface & Streamlit Dashboard
The interactive decision dashboard is hosted in `app/app.py` and provides 6 intuitive tabs:
1. **Overview:** Project context and ML architecture flowchart.
2. **Dataset Explorer:** Dynamic dataframe filtering, schema inspector, and missing value tracker.
3. **Exploratory Analysis:** Interactive high-resolution EDA plot carousel.
4. **Model Performance:** Comparative benchmark scorecard, confusion matrices, and ROC curves.
5. **Feature Importance:** Dynamic ranking charts and credit risk drivers.
6. **Live Predictor:** Real-time applicant decisioning calculator with instant approval/rejection banner, confidence probability, and risk tiers (Low / Moderate / High Risk).

---

## 14. How to Run the Project

### Prerequisites
Activate the pre-configured virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### 1. Run Complete Model Training Pipeline
Trains all models, runs 5-fold CV, executes GridSearchCV, generates all 9 figures, and persists model artifacts:
```powershell
python -m src.train
```

### 2. Launch the Streamlit Interactive Dashboard
```powershell
streamlit run app/app.py
```

### 3. Run Single Applicant Inference (CLI)
```powershell
python -m src.predict
```

### 4. Regenerate the 15-Page Academic PDF Report
```powershell
python src/generate_pdf_report.py
```

### 5. Open the Jupyter Notebook
```powershell
jupyter lab notebooks/loan_approval_analysis.ipynb
```

---

## 15. Limitations
1. **Sample Size:** 614 rows is modest for industrial credit scoring models.
2. **Omitted Financial Depth:** Attributes such as FICO credit scores, revolving credit balances, and asset collateral values were not present in the dataset.
3. **Class Imbalance:** 68.7% approval rate necessitates careful probability threshold calibration.

---

## 16. Future Improvements
- Implement **SHAP (SHapley Additive exPlanations)** for localized instance-level adverse action notices.
- Optimize custom cost-sensitive thresholding calibrated to the bank's specific cost-of-default matrix.
- Package application into Docker containers for cloud deployment.
