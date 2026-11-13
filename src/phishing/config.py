"""Paths and constants shared by the whole project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_DATASET = "shashwatwork/web-page-phishing-detection-dataset"
KAGGLE_FILE = RAW_DIR / "dataset_phishing.csv"
PHISHTANK_FILE = EXTERNAL_DIR / "phishtank_online_valid.csv"

URL_COL = "url"
TARGET_COL = "status"
LABELS = {"legitimate": 0, "phishing": 1}

# First column of each feature group, in the column order of the dataset.
# URL features: 56, content features: 24, external features: 7.
GROUP_STARTS = {
    "url": "length_url",
    "content": "nb_hyperlinks",
    "external": "whois_registered_domain",
}
GROUP_SIZES = {"url": 56, "content": 24, "external": 7}

RANDOM_STATE = 0
VAL_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 5

# Risk levels on the 1-10 score: [1, 4) low, [4, 7) medium, [7, 10] high.
RISK_LEVELS = (("low", 1.0), ("medium", 4.0), ("high", 7.0))
