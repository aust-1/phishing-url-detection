import warnings; warnings.filterwarnings("ignore")
from exp import *
from urlfe import registered_domain
from sklearn.pipeline import make_pipeline, make_union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from urllib.parse import urlsplit
tr,ytr,va,yva=load()
def host(s): 
    out=[]
    for u in s:
        try: out.append(urlsplit(u).hostname or "")
        except ValueError: out.append("")
    return out
def rest(s):
    out=[]
    for u in s:
        try: p=urlsplit(u); out.append(p.path+"?"+p.query)
        except ValueError: out.append(u)
    return out
cw=lambda ng=(1,5),mf=300000: TfidfVectorizer(analyzer="char_wb", ngram_range=ng, min_df=2, sublinear_tf=True, max_features=mf)
wd=lambda: TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z0-9]+", ngram_range=(1,2), min_df=2, sublinear_tf=True)
X,Xv=tr[["url"]],va[["url"]]
def ct(*parts): return ColumnTransformer([(f"p{i}",p,"url") for i,p in enumerate(parts)])
cands={
 "txt_cw15_C30": make_pipeline(ct(cw()), LogisticRegression(C=30,max_iter=5000)),
 "txt_cw15+wd_C10": make_pipeline(ct(cw(),wd()), LogisticRegression(C=10,max_iter=5000)),
 "txt_cw15+wd_C30": make_pipeline(ct(cw(),wd()), LogisticRegression(C=30,max_iter=5000)),
 "txt_hostpath_C10": make_pipeline(ct(make_pipeline(FunctionTransformer(host),cw((1,5))), make_pipeline(FunctionTransformer(rest),cw((1,5))), wd()), LogisticRegression(C=10,max_iter=5000)),
 "txt_cw36_C10": make_pipeline(ct(cw((3,6),500000)), LogisticRegression(C=10,max_iter=5000)),
 "txt_svc_cw15+wd": make_pipeline(ct(cw(),wd()), CalibratedClassifierCV(LinearSVC(C=0.5), cv=5, method="sigmoid")),
}
for n,m in cands.items():
    run(n, m, X, ytr, Xv, yva, note="r4 text")
    run(n+"|gcv", m, X, ytr, Xv, yva, note="r4 text", groups=tr["dom"], save=False)
