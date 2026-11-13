import warnings; warnings.filterwarnings("ignore")
from exp import *
from lightgbm import LGBMClassifier
tr,ytr,va,yva=load()
lg=LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=4, random_state=0)
fsets={"url":URL_F,"url+content":URL_F+CONTENT_F,"url+external":URL_F+EXTERNAL_F,"portable":PORTABLE_F,"content+external":CONTENT_F+EXTERNAL_F}
for fs,cols in fsets.items(): run(f"lgbm@{fs}", lg, tr[cols], ytr, va[cols], yva, note="r1b")
# domain-grouped CV for reference
run("lgbm@all|gcv", lg, tr[ALL_F], ytr, va[ALL_F], yva, note="r1b grouped", groups=tr["dom"])
