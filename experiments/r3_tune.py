"""Round 3: Optuna tuning (5-fold CV log loss on train only). Usage: r3_tune.py MODEL FSET MINUTES"""
import sys, json, warnings, time; warnings.filterwarnings("ignore")
import optuna
from exp import *
from urlfe import extract_extra
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

model, fset, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
tr, ytr, va, yva = load()
from urlfe import OOFProba, tfidf_lr
base = fset.split("+")[0]
cols = {"all": ALL_F, "url": URL_F, "portable": PORTABLE_F}[base]
X = tr[cols].reset_index(drop=True)
if "+extra" in fset:
    X = pd.concat([X, extract_extra(tr["url"])], axis=1)
if "+tfidf" in fset:
    cache = OUT / "tfidf_oof_train.npy"
    if not cache.exists():
        np.save(cache, OOFProba(tfidf_lr(), by_domain=True).fit_transform(tr[["url"]], ytr).ravel())
    X["tfidf_p"] = np.load(cache)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, ytr))  # other seed than eval

def make(trial):
    if model == "lgbm":
        return LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 200, 2000, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            verbose=-1, n_jobs=4, random_state=0)
    if model == "xgb":
        return XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 200, 2000, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30, log=True),
            n_jobs=4, random_state=0, tree_method="hist")
    if model == "cat":
        return CatBoostClassifier(iterations=trial.suggest_int("iterations", 300, 3000, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0, 2),
            verbose=0, thread_count=4, random_seed=0)
    if model == "et":
        return ExtraTreesClassifier(n_estimators=600, max_features=trial.suggest_float("max_features", 0.05, 0.8),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5), criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            n_jobs=4, random_state=0)

def objective(trial):
    m = make(trial); p = np.zeros(len(ytr))
    for k, (a, b) in enumerate(folds):
        m.fit(X.iloc[a], ytr[a]); p[b] = m.predict_proba(X.iloc[b])[:, 1]
        trial.report(log_loss(ytr[b], np.clip(p[b], 1e-6, 1 - 1e-6)), k)
        if trial.should_prune(): raise optuna.TrialPruned()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    trial.set_user_attr("acc", float(((p >= .5) == ytr).mean()))
    return log_loss(ytr, pc)

optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                         pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
st.optimize(objective, timeout=minutes * 60, gc_after_trial=True)
b = st.best_trial
print(f"{model}@{fset}: {len(st.trials)} trials, best logloss {b.value:.4f} acc {b.user_attrs.get('acc')}")
print(json.dumps(b.params))
(OUT / f"best_{model}@{fset}.json").write_text(json.dumps(b.params, indent=1))
st.trials_dataframe().to_csv(OUT / f"trials_{model}@{fset}.csv", index=False)
