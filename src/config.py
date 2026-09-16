"""Central configuration. All paths and constants live here."""
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR    = REPORT_DIR / "figures"
DB_DIR     = BASE_DIR / "database"
SHOT_DIR   = BASE_DIR / "screenshots"
TEST_DIR   = BASE_DIR / "tests"

ALL_DIRS = [DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR, DB_DIR, SHOT_DIR, TEST_DIR]

DATASET_PATH   = DATA_DIR / "Phishing_Email.csv"
DB_PATH        = DB_DIR / "phishing_detector.db"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "phishing_classifier.joblib"
METADATA_PATH   = MODEL_DIR / "model_metadata.joblib"

TEXT_COL   = "Email Text"
TARGET_COL = "Email Type"
CLEAN_COL  = "Cleaned_Email_Text"
INDEX_COL  = "Unnamed: 0"

SAFE_LABEL     = "Safe Email"
PHISHING_LABEL = "Phishing Email"
LABEL_MAP      = {SAFE_LABEL: 0, PHISHING_LABEL: 1}
INVERSE_MAP    = {0: SAFE_LABEL, 1: PHISHING_LABEL}

RANDOM_STATE = 42
TEST_SIZE    = 0.20
PREPROCESSING_VERSION = "v1.0"
MODEL_NAME   = "LogisticRegression + TF-IDF (1,2)-gram"
MAX_STORED_TEXT_CHARS = 500   # limit sensitive data stored in DB
