from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

RAW_DATA_FILE = RAW_DATA_DIR / "loan_data.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_loan_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Column definitions
ID_COL = "Loan_ID"
TARGET_COL = "Loan_Status"

NUMERICAL_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Credit_History",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
