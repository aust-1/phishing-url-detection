"""Round 2: URL-derived features (extra hand-crafted, char n-gram TF-IDF)."""
import sys, warnings; warnings.filterwarnings("ignore")
from exp import *
from urlfe import UrlExtra
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

tr, ytr, va, yva = load()
lgbm = lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=4, random_state=0)

def with_extra(cols, model):
    ct = ColumnTransformer([("base", "passthrough", cols), ("extra", UrlExtra(), ["url"])])
    return make_pipeline(ct, model)

cols_tr = ALL_F + ["url"]
step = sys.argv[1]
if step == "extra":
    run("lgbm@all+extra", with_extra(ALL_F, lgbm()), tr[cols_tr], ytr, va[cols_tr], yva, note="r2")
    run("lgbm@url+extra", with_extra(URL_F, lgbm()), tr[URL_F+["url"]], ytr, va[URL_F+["url"]], yva, note="r2")
    run("lgbm@portable+extra", with_extra(PORTABLE_F, lgbm()), tr[PORTABLE_F+["url"]], ytr, va[PORTABLE_F+["url"]], yva, note="r2")
    run("cat@all+extra", with_extra(ALL_F, CatBoostClassifier(iterations=1000, verbose=0, thread_count=4, random_seed=0)), tr[cols_tr], ytr, va[cols_tr], yva, note="r2")
if step == "tfidf":
    for ng, an in [((1, 5), "char_wb"), ((2, 6), "char"), ((1, 2), "word")]:
        vec = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=2, sublinear_tf=True, lowercase=True,
                              token_pattern=r"[A-Za-z0-9]+" if an == "word" else None, max_features=300000)
        for C in (3, 10):
            m = make_pipeline(ColumnTransformer([("t", vec, "url")]), LogisticRegression(C=C, max_iter=3000))
            run(f"tfidf_{an}{ng[0]}{ng[1]}_C{C}@url", m, tr[["url"]], ytr, va[["url"]], yva, note="r2")
