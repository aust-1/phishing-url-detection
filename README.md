# Phishing URL Detection and Risk Scoring

ESILV A4 Machine Learning project (2026-2027).

Team: Alexandra de Parcevaux, Camille Leyris, Charlotte Huon de Kermadec, Eliott Roussille, Eloi Lacour.

Goal: classify URLs as legitimate (0) or phishing (1) from lexical and domain
features, and give each URL a risk score from 1 to 10:

```plaintext
risk score = 1 + 9 * P(phishing)
low: [1, 4)    medium: [4, 7)    high: [7, 10]
```

## Repository structure

```plaintext
.
├── data/
│   ├── raw/              # dataset_phishing.csv (Kaggle)
│   ├── external/         # phishtank_online_valid.csv (PhishTank)
│   ├── processed/        # splits and generated files
│   └── README.md         # how to get the data
├── notebooks/
│   ├── v1.ipynb              # first look at the data
│   └── v2_meilleur_modele.ipynb  # best model: comparison, final test, PhishTank
├── reports/
│   └── figures/          # generated figures
├── scripts/
│   ├── make_splits.py      # clean + stratified 70/15/15 split
│   ├── eda.py              # first EDA on the train split
│   ├── check_features.py   # compare our URL features with the dataset
│   └── fetch_phishtank.py  # download PhishTank + compute URL features
├── src/phishing/
│   ├── config.py         # paths and constants
│   ├── data.py           # load, clean, feature groups, split
│   ├── features.py       # lexical features from a raw URL
│   ├── url_text.py       # extra URL features, char TF-IDF model, out-of-fold stacking
│   ├── phishtank.py      # PhishTank download and load
│   ├── risk.py           # risk score and risk level
│   └── evaluation.py     # metrics and figures
├── tests/
├── pyproject.toml
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
# 1. put dataset_phishing.csv in data/raw/ (see data/README.md)
python scripts/make_splits.py
python scripts/eda.py
python scripts/check_features.py

# 2. best model (about 15 min on 4 cores; RUN_BENCHMARK = False skips the comparison)
jupyter nbconvert --to notebook --execute --inplace notebooks/v2_meilleur_modele.ipynb

# 3. optional, PhishTank test set
export PHISHTANK_USER_AGENT=phishtank/<your-username>
export PHISHTANK_APP_KEY=<key>
python scripts/fetch_phishtank.py
```

## Rules

- The test split is used only once, for the final evaluation.
- Scaling and feature selection are fitted inside a scikit-learn `Pipeline`,
  only on training folds.
- The `url` column is never given to a model as it is: only the transformers of
  `phishing.url_text` read it, inside the pipeline.
- All models are compared with the baseline using `phishing.evaluation`.

## Development

```bash
pytest
ruff check .
ruff format .
```

## Data sources

- Hannousse A., Yahiouche S. (2021). Towards benchmark datasets for machine learning
  based website phishing detection: An experimental study. Engineering Applications
  of Artificial Intelligence, 104, 104347.
- PhishTank: <https://phishtank.net>
