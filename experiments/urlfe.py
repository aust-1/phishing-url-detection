"""Extra features computed from the raw URL string (no network access)."""

import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

_EX = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account", "update",
    "secure", "security", "bank", "confirm", "password", "webscr", "ebayisapi", "wallet",
    "support", "billing", "invoice", "unlock", "suspend", "recover", "auth", "token",
    "session", "admin", "wp-", "includes", "dropbox", "docusign", "office", "outlook",
    "microsoft", "paypal", "apple", "amazon", "netflix", "free", "bonus", "gift", "lucky",
]
FREE_HOSTS = [
    "000webhostapp", "weebly", "wix", "firebaseapp", "web.app", "appspot", "blogspot",
    "sites.google", "github.io", "herokuapp", "netlify", "vercel", "glitch.me", "codesandbox",
    "repl.co", "sharepoint", "ipfs", "webcindario", "wordpress", "tumblr", "duckdns", "ngrok",
    "workers.dev", "pages.dev", "azurewebsites", "cloudfront", "r2.dev", "myshopify",
    "square.site", "godaddysites", "yolasite", "jimdo", "bit.ly", "tinyurl",
]
SHORTENERS = ["bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co", "is.gd", "buff.ly", "rb.gy",
              "cutt.ly", "shorturl", "tiny.cc"]
COMMON_TLDS = ["com", "org", "net", "edu", "gov", "de", "uk", "fr", "io", "ru", "br", "in",
               "info", "co", "jp", "it", "nl", "au", "ca", "es", "pl", "cn", "me", "xyz", "top",
               "tk", "ml", "ga", "cf", "gq", "app", "dev", "site", "online", "club", "biz"]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _max_run(s: str, pred) -> int:
    best = cur = 0
    for ch in s:
        cur = cur + 1 if pred(ch) else 0
        best = max(best, cur)
    return best


def url_extra(url: str) -> dict:
    u = str(url)
    low = u.lower()
    try:
        p = urlsplit(u)
        host = (p.hostname or "").lower()
        path, query = p.path, p.query
    except ValueError:
        host, path, query = "", "", ""
    ext = _EX(host or u)
    dom, sub, suf = ext.domain, ext.subdomain, ext.suffix
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    host_tokens = [t for t in re.split(r"[.\-]", host) if t]
    vowels = sum(c in "aeiou" for c in dom)
    f = {
        "x_len_dom": len(dom),
        "x_len_sub": len(sub),
        "x_len_path": len(path),
        "x_len_query": len(query),
        "x_n_params": len([q for q in query.split("&") if q]),
        "x_path_depth": len([s for s in path.split("/") if s]),
        "x_ent_url": _entropy(low),
        "x_ent_host": _entropy(host),
        "x_ent_dom": _entropy(dom),
        "x_digits_dom": sum(c.isdigit() for c in dom),
        "x_vowel_ratio_dom": vowels / len(dom) if dom else 0.0,
        "x_max_cons_run_dom": _max_run(dom, lambda c: c.isalpha() and c not in "aeiou"),
        "x_max_digit_run": _max_run(low, str.isdigit),
        "x_n_tokens": len(tokens),
        "x_n_host_tokens": len(host_tokens),
        "x_longest_token": max((len(t) for t in tokens), default=0),
        "x_upper_ratio": sum(c.isupper() for c in u) / len(u) if u else 0.0,
        "x_hex_ratio": len(re.findall(r"%[0-9a-f]{2}", low)) / max(len(u), 1),
        "x_susp_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "x_susp_words_host": sum(w in host for w in SUSPICIOUS_WORDS),
        "x_free_host": int(any(h in host for h in FREE_HOSTS)),
        "x_shortener": int(any(host == s or host.endswith("." + s) for s in SHORTENERS)),
        "x_https": int(low.startswith("https")),
        "x_ext_php": int(path.lower().endswith(".php")),
        "x_ext_html": int(bool(re.search(r"\.html?$", path.lower()))),
        "x_ext_exe": int(bool(re.search(r"\.(exe|zip|apk|rar|js)$", path.lower()))),
        "x_email_in_url": int(bool(re.search(r"[\w.]+@[\w.]+\.\w+", low))),
        "x_www": int(host.startswith("www.")),
        "x_tld_len": len(suf),
        "x_n_sub_levels": len([s for s in sub.split(".") if s]),
        "x_trailing_slash": int(u.endswith("/")),
        "x_path_only_root": int(path in ("", "/") and not query),
        "x_tld_in_path_tok": int(any(t in COMMON_TLDS[:6] for t in re.split(r"[^a-z]+", path.lower()))),
        "x_dom_is_ip": int(bool(re.fullmatch(r"[\d.]+", host))),
    }
    tld = suf.split(".")[-1] if suf else ""
    for t in COMMON_TLDS:
        f[f"x_tld_{t}"] = int(tld == t)
    return f


def extract_extra(urls) -> pd.DataFrame:
    return pd.DataFrame([url_extra(u) for u in urls])


class UrlExtra(BaseEstimator, TransformerMixin):
    """Transformer: takes a DataFrame with a `url` column, returns extra features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_extra(X["url"]).to_numpy(float)


def registered_domain(url: str) -> str:
    try:
        host = urlsplit(str(url)).hostname or str(url)
    except ValueError:
        host = str(url)
    r = _EX(host)
    return r.top_domain_under_public_suffix or r.domain or host


class OOFProba(BaseEstimator, TransformerMixin):
    """Stacking feature: probability of an inner model.

    fit_transform returns out-of-fold probabilities (the row is never seen by the
    model that scores it); transform uses the model refitted on all the data.
    """

    def __init__(self, estimator, n_splits=5, random_state=0, by_domain=False):
        self.estimator = estimator
        self.n_splits = n_splits
        self.random_state = random_state
        self.by_domain = by_domain

    def fit(self, X, y):
        from sklearn.base import clone
        self.model_ = clone(self.estimator).fit(X, y)
        return self

    def fit_transform(self, X, y=None, **kw):
        from sklearn.base import clone
        from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                             cross_val_predict)
        groups = None
        if self.by_domain:  # inner folds never share a registered domain
            groups = X["url"].map(registered_domain).to_numpy()
            cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        p = cross_val_predict(clone(self.estimator), X, y, cv=cv, groups=groups,
                              method="predict_proba")[:, 1]
        self.fit(X, y)
        return p.reshape(-1, 1)

    def transform(self, X):
        return self.model_.predict_proba(X)[:, 1].reshape(-1, 1)


def tfidf_lr(C=10.0):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                          max_features=300000)
    return make_pipeline(ColumnTransformer([("t", vec, "url")]), LogisticRegression(C=C, max_iter=3000))
