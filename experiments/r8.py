import sys, warnings; warnings.filterwarnings("ignore")
from exp import *
from models import gbm, features
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
tr,ytr,va,yva=load()
cols=ALL_F+["url"]; X,Xv=tr[cols],va[cols]
et=lambda: ExtraTreesClassifier(800, max_features=0.3, n_jobs=4, random_state=0)
c={"R_lgbm": gbm("lgbm"),
   "R_vote3": VotingClassifier([(n,gbm(n)) for n in ("lgbm","xgb","cat")], voting="soft"),
   "R_vote4et": VotingClassifier([(n,gbm(n)) for n in ("lgbm","xgb","cat")]+[("et",et())], voting="soft")}
for n,m in c.items():
    run(n, make_pipeline(features(), m), X,ytr,Xv,yva, seeds=(1,2,3), note="r8 repeated 3x5")
run("R_vote4et|gcv", make_pipeline(features(), c["R_vote4et"]), X,ytr,Xv,yva, note="r8", groups=tr["dom"])
