"""Experiment harness: CV on train (OOF probas saved), fit on train, eval on val.

The test split is never read here.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from phishing.data import load_split  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
RESULTS = OUT / "results.csv"

GROUPS = json.loads((REPO / "data/processed/feature_groups.json").read_text())
CONSTANT = ["nb_or", "ratio_nullHyperlinks", "ratio_intRedirection", "ratio_intErrors",
            "submit_email", "sfh"]
URL_F = [c for c in GROUPS["url"] if c not in CONSTANT]
CONTENT_F = [c for c in GROUPS["content"] if c not in CONSTANT]
EXTERNAL_F = GROUPS["external"]
ALL_F = URL_F + CONTENT_F + EXTERNAL_F
PORTABLE_F = json.loads((REPO / "data/processed/portable_features.json").read_text())


VAL_UNSEEN = None


def load():
    """Train and val, with the registered domain of each URL in a `dom` column."""
    global VAL_UNSEEN
    from urlfe import registered_domain

    tr, va = load_split("train"), load_split("val")
    tr["dom"] = tr["url"].map(registered_domain)
    va["dom"] = va["url"].map(registered_domain)
    VAL_UNSEEN = ~va["dom"].isin(set(tr["dom"])).to_numpy()
    return tr, tr["status"].to_numpy(), va, va["status"].to_numpy()


def metrics(y, p, thr=0.5):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = (p >= thr).astype(int)
    fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    return {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "fpr": fp / (fp + tn),
    }


def oof(model, X, y, seeds=(0,), n_splits=5, groups=None):
    """Out-of-fold probabilities, averaged over the repeats.

    With `groups`, folds never share a group (registered domain)."""
    P = np.zeros((len(seeds), len(y)))
    for i, s in enumerate(seeds):
        cv = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=s) if groups is not None
              else StratifiedKFold(n_splits, shuffle=True, random_state=s))
        for tr_idx, te_idx in cv.split(X, y, groups):
            m = clone(model).fit(X.iloc[tr_idx], y[tr_idx])
            P[i, te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]
    return P


def run(name, model, X_tr, y_tr, X_va, y_va, seeds=(0,), note="", save=True, groups=None):
    t = time.time()
    P = oof(model, X_tr, y_tr, seeds, groups=groups)
    cv = [metrics(y_tr, p) for p in P]
    cvm = {k: np.mean([c[k] for c in cv]) for k in cv[0]}
    m = clone(model).fit(X_tr, y_tr)
    pv = m.predict_proba(X_va)[:, 1]
    vm = metrics(y_va, pv)
    if VAL_UNSEEN is not None:  # val rows whose registered domain is not in train
        um = metrics(y_va[VAL_UNSEEN], pv[VAL_UNSEEN])
        vm.update({"unseen_acc": um["acc"], "unseen_auc": um["auc"]})
    dt = time.time() - t
    row = {"name": name, **{f"cv_{k}": v for k, v in cvm.items()},
           **{f"val_{k}": v for k, v in vm.items()}, "sec": round(dt, 1),
           "n_feat": X_tr.shape[1], "note": note}
    if save:
        np.save(OUT / f"oof_{name}.npy", P.mean(0))
        np.save(OUT / f"val_{name}.npy", pv)
        old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
        old = old[old["name"] != name] if len(old) else old
        pd.concat([old, pd.DataFrame([row])]).to_csv(RESULTS, index=False)
    print(f"{name:<40} cv acc {cvm['acc']:.4f} auc {cvm['auc']:.4f} brier {cvm['brier']:.4f} | "
          f"val acc {vm['acc']:.4f} auc {vm['auc']:.4f} brier {vm['brier']:.4f} | "
          f"unseen acc {vm.get('unseen_acc', float('nan')):.4f} | {dt:.0f}s",
          flush=True)
    return row, m
