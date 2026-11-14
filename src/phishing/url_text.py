"""Features computed from the raw URL string, on top of the dataset columns.

- `extra_features`: hand-crafted lexical features (entropy, path depth, suspicious
  words, free hosting, TLD, ...), computed without any network access.
- `char_tfidf_model`: logistic regression on character n-grams of the URL.
- `OutOfFoldProba`: turns a model into one stacking feature, its out-of-fold
  probability, so the row being scored was never seen by the model.

The `url` column is only read by these transformers inside a scikit-learn
`Pipeline`, it is never given to a model as it is.
"""

import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline

from phishing import config

# Offline: uses the public suffix list shipped with tldextract.
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account", "update",
    "secure", "security", "bank", "confirm", "password", "webscr", "ebayisapi", "wallet",
    "support", "billing", "invoice", "unlock", "suspend", "recover", "auth", "token",
    "session", "admin", "wp-", "includes", "dropbox", "docusign", "office", "outlook",
    "microsoft", "paypal", "apple", "amazon", "netflix", "free", "bonus", "gift", "lucky",
)  # fmt: skip
FREE_HOSTS = (
    "000webhostapp", "weebly", "wix", "firebaseapp", "web.app", "appspot", "blogspot",
    "sites.google", "github.io", "herokuapp", "netlify", "vercel", "glitch.me", "codesandbox",
    "repl.co", "sharepoint", "ipfs", "webcindario", "wordpress", "tumblr", "duckdns", "ngrok",
    "workers.dev", "pages.dev", "azurewebsites", "cloudfront", "r2.dev", "myshopify",
    "square.site", "godaddysites", "yolasite", "jimdo", "bit.ly", "tinyurl",
)  # fmt: skip
SHORTENERS = (
    "bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co", "is.gd", "buff.ly", "rb.gy", "cutt.ly",
    "shorturl", "tiny.cc",
)  # fmt: skip
COMMON_TLDS = (
    "com", "org", "net", "edu", "gov", "de", "uk", "fr", "io", "ru", "br", "in", "info", "co",
    "jp", "it", "nl", "au", "ca", "es", "pl", "cn", "me", "xyz", "top", "tk", "ml", "ga", "cf",
    "gq", "app", "dev", "site", "online", "club", "biz",
)  # fmt: skip


def _split(url: str) -> tuple[str, str, str]:
    """Host (lower case), path and query of a URL; empty strings if it cannot be parsed."""
    try:
        parts = urlsplit(url)
        return (parts.hostname or "").lower(), parts.path, parts.query
    except ValueError:
        return "", "", ""


def registered_domain(url: str) -> str:
    """Registered domain, for example `paypal.com` for `https://www.paypal.com/fr`."""
    host = _split(str(url))[0] or str(url)
    parts = _TLD_EXTRACT(host)
    return parts.top_domain_under_public_suffix or parts.domain or host


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _longest_run(text: str, keep) -> int:
    best = current = 0
    for ch in text:
        current = current + 1 if keep(ch) else 0
        best = max(best, current)
    return best


def url_extra_features(url: str) -> dict[str, float]:
    """Extra lexical features of one URL. Names start with `x_`."""
    url = str(url)
    low = url.lower()
    host, path, query = _split(url)
    parts = _TLD_EXTRACT(host or url)
    dom, sub, suffix = parts.domain, parts.subdomain, parts.suffix
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    lpath = path.lower()

    features = {
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
        "x_vowel_ratio_dom": sum(c in "aeiou" for c in dom) / len(dom) if dom else 0.0,
        "x_max_cons_run_dom": _longest_run(dom, lambda c: c.isalpha() and c not in "aeiou"),
        "x_max_digit_run": _longest_run(low, str.isdigit),
        "x_n_tokens": len(tokens),
        "x_n_host_tokens": len([t for t in re.split(r"[.\-]", host) if t]),
        "x_longest_token": max((len(t) for t in tokens), default=0),
        "x_upper_ratio": sum(c.isupper() for c in url) / len(url) if url else 0.0,
        "x_hex_ratio": len(re.findall(r"%[0-9a-f]{2}", low)) / max(len(url), 1),
        "x_susp_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "x_susp_words_host": sum(w in host for w in SUSPICIOUS_WORDS),
        "x_free_host": int(any(h in host for h in FREE_HOSTS)),
        "x_shortener": int(any(host == s or host.endswith("." + s) for s in SHORTENERS)),
        "x_https": int(low.startswith("https")),
        "x_ext_php": int(lpath.endswith(".php")),
        "x_ext_html": int(bool(re.search(r"\.html?$", lpath))),
        "x_ext_exe": int(bool(re.search(r"\.(exe|zip|apk|rar|js)$", lpath))),
        "x_email_in_url": int(bool(re.search(r"[\w.]+@[\w.]+\.\w+", low))),
        "x_www": int(host.startswith("www.")),
        "x_tld_len": len(suffix),
        "x_n_sub_levels": len([s for s in sub.split(".") if s]),
        "x_trailing_slash": int(url.endswith("/")),
        "x_path_only_root": int(path in ("", "/") and not query),
        "x_tld_in_path_tok": int(any(t in COMMON_TLDS[:6] for t in re.split(r"[^a-z]+", lpath))),
        "x_dom_is_ip": int(bool(re.fullmatch(r"[\d.]+", host))),
    }
    tld = suffix.split(".")[-1] if suffix else ""
    features.update({f"x_tld_{t}": int(tld == t) for t in COMMON_TLDS})
    return features


EXTRA_FEATURE_NAMES = tuple(url_extra_features("http://example.com"))


def extra_features(urls: pd.Series) -> pd.DataFrame:
    """Extra lexical features of a series of URLs."""
    rows = [url_extra_features(u) for u in urls]
    return pd.DataFrame(rows, columns=list(EXTRA_FEATURE_NAMES), index=urls.index)


class ExtraFeatures(BaseEstimator, TransformerMixin):
    """Pipeline step: DataFrame with a `url` column -> extra lexical features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extra_features(X[config.URL_COL]).to_numpy(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(EXTRA_FEATURE_NAMES, dtype=object)


def char_tfidf_model(C: float = 10.0) -> Pipeline:
    """Logistic regression on TF-IDF character 1- to 5-grams of the URL."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True, max_features=300_000
    )
    return make_pipeline(
        ColumnTransformer([("tfidf", vectorizer, config.URL_COL)]),
        LogisticRegression(C=C, max_iter=3000),
    )


class OutOfFoldProba(BaseEstimator, TransformerMixin):
    """Stacking feature: the phishing probability given by an inner model.

    `fit_transform` (training) returns out-of-fold probabilities: each row is scored
    by a model that did not see it. With `by_domain=True` the inner folds never share
    a registered domain, so the feature is as reliable on training rows as on URLs of
    unknown domains. `transform` (new data) uses the model refitted on all the rows.
    """

    def __init__(self, estimator, n_splits: int = 5, by_domain: bool = True, random_state=0):
        self.estimator = estimator
        self.n_splits = n_splits
        self.by_domain = by_domain
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = clone(self.estimator).fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if self.by_domain:
            groups = X[config.URL_COL].map(registered_domain).to_numpy()
            cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            groups = None
            cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        proba = cross_val_predict(
            clone(self.estimator), X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
        self.fit(X, y)
        return proba.reshape(-1, 1)

    def transform(self, X):
        return self.model_.predict_proba(X)[:, 1].reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["oof_proba"], dtype=object)
