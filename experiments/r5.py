import warnings; warnings.filterwarnings("ignore")
from exp import *
from urlfe import UrlExtra, OOFProba, text_model
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
tr,ytr,va,yva=load()
lg=LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=4, random_state=0)
cols=ALL_F+["url"]
m=make_pipeline(ColumnTransformer([("base","passthrough",ALL_F),("extra",UrlExtra(),["url"]),("txt",OOFProba(text_model(),by_domain=True),["url"])]), lg)
run("lgbm@all+extra+txtHP", m, tr[cols], ytr, va[cols], yva, note="r5")
run("lgbm@all+extra+txtHP|gcv", m, tr[cols], ytr, va[cols], yva, note="r5", groups=tr["dom"])
