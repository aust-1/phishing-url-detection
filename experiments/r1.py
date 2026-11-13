import warnings; warnings.filterwarnings("ignore")
from exp import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

tr,ytr,va,yva=load()
models={
 "dummy": DummyClassifier(),
 "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
 "logreg_qt": make_pipeline(QuantileTransformer(output_distribution="normal"), LogisticRegression(max_iter=3000)),
 "gnb": make_pipeline(QuantileTransformer(output_distribution="normal"), GaussianNB()),
 "knn": make_pipeline(QuantileTransformer(output_distribution="normal"), KNeighborsClassifier(15, weights="distance")),
 "tree": DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
 "rf": RandomForestClassifier(500, n_jobs=4, random_state=0),
 "et": ExtraTreesClassifier(500, n_jobs=4, random_state=0),
 "hgb": HistGradientBoostingClassifier(random_state=0),
 "svc_rbf": make_pipeline(QuantileTransformer(output_distribution="normal"), SVC(C=3, probability=True, random_state=0)),
 "mlp": make_pipeline(QuantileTransformer(output_distribution="normal"), MLPClassifier((128,64), alpha=1e-3, max_iter=500, early_stopping=True, random_state=0)),
 "lgbm": LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=4, random_state=0),
 "xgb": XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, n_jobs=4, random_state=0),
 "cat": CatBoostClassifier(iterations=1000, verbose=0, thread_count=4, random_seed=0),
}
import sys
fsets={"all":ALL_F,"url":URL_F,"url+content":URL_F+CONTENT_F,"url+external":URL_F+EXTERNAL_F,"portable":PORTABLE_F}
which=sys.argv[1:] or list(fsets)
for fs in which:
    cols=fsets[fs]
    for n,m in models.items():
        if fs!="all" and n in ("dummy","gnb","tree","knn"): continue
        run(f"{n}@{fs}", m, tr[cols], ytr, va[cols], yva, note="r1")
