import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from phishing.url_text import (
    EXTRA_FEATURE_NAMES,
    ExtraFeatures,
    OutOfFoldProba,
    extra_features,
    registered_domain,
    url_extra_features,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.paypal.com/fr/signin", "paypal.com"),
        ("http://a.b.example.co.uk/x", "example.co.uk"),
        ("http://e710z0ear.du.r.appspot.com/c:/users", "appspot.com"),
        ("http://192.168.1.10/login", "192.168.1.10"),
    ],
)
def test_registered_domain(url, expected):
    assert registered_domain(url) == expected


def test_extra_features_values():
    f = url_extra_features("http://login.secure-bank.tk/a/b/verify.php?id=12345&x=1")
    assert set(f) == set(EXTRA_FEATURE_NAMES)
    assert f["x_path_depth"] == 3
    assert f["x_n_params"] == 2
    assert f["x_max_digit_run"] == 5
    assert f["x_ext_php"] == 1
    assert f["x_https"] == 0
    assert f["x_tld_tk"] == 1 and f["x_tld_com"] == 0
    assert f["x_susp_words"] >= 3  # login, secure, bank, verify


def test_extra_features_bad_url_does_not_fail():
    f = url_extra_features("http://[invalid")
    assert f["x_len_dom"] >= 0


def test_extra_features_frame():
    urls = pd.Series(["https://example.com", "http://x.org/a?b=c"], index=[5, 9])
    df = extra_features(urls)
    assert list(df.index) == [5, 9]
    assert list(df.columns) == list(EXTRA_FEATURE_NAMES)
    out = ExtraFeatures().fit_transform(pd.DataFrame({"url": urls}))
    assert out.shape == (2, len(EXTRA_FEATURE_NAMES))


def test_out_of_fold_proba_is_not_fitted_on_the_scored_row():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"url": [f"http://site{i % 40}.com/p" for i in range(n)], "v": rng.random(n)})
    y = (X["v"] > 0.5).astype(int).to_numpy()
    inner = make_pipeline(ColumnTransformer([("v", "passthrough", ["v"])]), LogisticRegression())
    step = OutOfFoldProba(inner, n_splits=5, by_domain=True)
    oof = step.fit_transform(X, y)
    assert oof.shape == (n, 1)
    assert np.all((oof >= 0) & (oof <= 1))
    # transform uses the model refitted on all rows
    full = step.transform(X)
    assert not np.allclose(oof, full)
