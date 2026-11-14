import sys, warnings; warnings.filterwarnings("ignore")
from exp import *
from models import gbm, features
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
tr,ytr,va,yva=load()
cols=ALL_F+["url"]; X,Xv=tr[cols],va[cols]
step=sys.argv[1]
if step=="single":
    for n in ("lgbm","xgb","cat"):
        run(f"T_{n}", make_pipeline(features(), gbm(n)), X,ytr,Xv,yva, note="r6 tuned")
if step=="vote":
    v=VotingClassifier([(n,gbm(n)) for n in ("lgbm","xgb","cat")], voting="soft")
    run("T_vote3", make_pipeline(features(), v), X,ytr,Xv,yva, note="r6 tuned")
    run("T_vote3|gcv", make_pipeline(features(), v), X,ytr,Xv,yva, note="r6 tuned", groups=tr["dom"])
