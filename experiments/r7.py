import sys, warnings; warnings.filterwarnings("ignore")
from exp import *
from models import gbm, features
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from phishing.url_text import registered_domain, ExtraFeatures, OutOfFoldProba, char_tfidf_model
tr,ytr,va,yva=load()
cols=ALL_F+["url"]; X,Xv=tr[cols],va[cols]
et=lambda: ExtraTreesClassifier(800, max_features=0.3, min_samples_leaf=1, n_jobs=4, random_state=0)
step=sys.argv[1]
if step=="urlonly":
    c2=PORTABLE_F+["url"]
    m=make_pipeline(features(PORTABLE_F), gbm("lgbm"))
    run("T_lgbm@portable+extra+tfidf", m, tr[c2],ytr,va[c2],yva, note="r7 url-only")
    run("T_lgbm@portable+extra+tfidf|gcv", m, tr[c2],ytr,va[c2],yva, note="r7 url-only", groups=tr["dom"])
    c3=URL_F+["url"]
    m=make_pipeline(features(URL_F), gbm("lgbm"))
    run("T_lgbm@url+extra+tfidf", m, tr[c3],ytr,va[c3],yva, note="r7 url-only")
if step=="vote4":
    v=VotingClassifier([(n,gbm(n)) for n in ("lgbm","xgb","cat")]+[("et",et())], voting="soft")
    run("T_vote4et", make_pipeline(features(), v), X,ytr,Xv,yva, note="r7")
if step=="stack":
    s=StackingClassifier([(n,gbm(n)) for n in ("lgbm","xgb","cat")]+[("et",et())], final_estimator=LogisticRegression(), cv=5, n_jobs=1)
    run("T_stackLR", make_pipeline(features(), s), X,ytr,Xv,yva, note="r7")
if step=="domte":
    class Dom(BaseEstimator, TransformerMixin):
        def fit(self,X,y=None): return self
        def transform(self,X): return X["url"].map(registered_domain).to_frame("dom")
    ct=ColumnTransformer([("base","passthrough",ALL_F),("extra",ExtraFeatures(),["url"]),
        ("tfidf",OutOfFoldProba(char_tfidf_model()),["url"]),
        ("domte",make_pipeline(Dom(),TargetEncoder(target_type="binary",random_state=0)),["url"])])
    run("T_lgbm+domTE", make_pipeline(ct,gbm("lgbm")), X,ytr,Xv,yva, note="r7 domain target enc")
    run("T_lgbm+domTE|gcv", make_pipeline(ct,gbm("lgbm")), X,ytr,Xv,yva, note="r7", groups=tr["dom"])
