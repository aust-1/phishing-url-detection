"""Tuned model factories (params from out/best_*.json) and the full stacked pipeline."""
import json
from exp import OUT, ALL_F, PORTABLE_F
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import sys
from phishing.url_text import ExtraFeatures, OutOfFoldProba, char_tfidf_model

def params(name, fset="all+extra+tfidf"):
    return json.loads((OUT / f"best_{name}@{fset}.json").read_text())

def gbm(name, seed=0, fset="all+extra+tfidf"):
    p = params(name, fset)
    if name == "lgbm": return LGBMClassifier(**p, subsample_freq=1, verbose=-1, n_jobs=4, random_state=seed)
    if name == "xgb": return XGBClassifier(**p, n_jobs=4, random_state=seed, tree_method="hist")
    if name == "cat": return CatBoostClassifier(**p, verbose=0, thread_count=4, random_seed=seed)

def features(base=ALL_F, extra=True, tfidf=True):
    parts = [("base", "passthrough", list(base))]
    if extra: parts.append(("extra", ExtraFeatures(), ["url"]))
    if tfidf: parts.append(("tfidf", OutOfFoldProba(char_tfidf_model()), ["url"]))
    return ColumnTransformer(parts)
