import warnings; warnings.filterwarnings("ignore")
from exp import *
from urlfe import UrlExtra, OOFProba, tfidf_lr
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
tr,ytr,va,yva=load()
lg=LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=4, random_state=0)
cols=ALL_F+["url"]
ct=ColumnTransformer([("base","passthrough",ALL_F),("extra",UrlExtra(),["url"]),("tfidf",OOFProba(tfidf_lr()),["url"])])
run("lgbm@all+extra+tfidf", make_pipeline(ct, lg), tr[cols], ytr, va[cols], yva, note="r2b stack")
ct=ColumnTransformer([("base","passthrough",ALL_F),("tfidf",OOFProba(tfidf_lr()),["url"])])
run("lgbm@all+tfidf", make_pipeline(ct, lg), tr[cols], ytr, va[cols], yva, note="r2b stack")
