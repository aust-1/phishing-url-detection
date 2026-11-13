import pandas as pd
import pytest

from phishing.features import FEATURE_NAMES, extract, url_features


def test_all_features_present():
    assert set(url_features("http://example.com")) == set(FEATURE_NAMES)


def test_basic_counts():
    f = url_features("http://login-secure.example.com/a_b?x=1&y=2")
    assert f["length_url"] == len("http://login-secure.example.com/a_b?x=1&y=2")
    assert f["length_hostname"] == len("login-secure.example.com")
    assert f["nb_dots"] == 2
    assert f["nb_hyphens"] == 1
    assert f["nb_qm"] == 1
    assert f["nb_and"] == 1
    assert f["nb_eq"] == 2
    assert f["nb_underscore"] == 1
    assert f["nb_com"] == 1


@pytest.mark.parametrize(
    ("url", "expected"),
    [("http://192.168.1.10/login", 1), ("http://[::1]/", 1), ("https://example.com", 0)],
)
def test_ip(url, expected):
    assert url_features(url)["ip"] == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [("https://example.com", 0), ("http://example.com", 1)],
)
def test_https_token(url, expected):
    assert url_features(url)["https_token"] == expected


def test_double_slash_redirection():
    assert url_features("https://example.com/path")["nb_dslash"] == 0
    assert url_features("http://example.com//evil.com")["nb_dslash"] == 1


def test_port_and_punycode():
    assert url_features("http://example.com:8080/")["port"] == 1
    assert url_features("http://example.com:99999/")["port"] == 1
    assert url_features("http://xn--pple-43d.com")["punycode"] == 1


def test_digit_ratio_empty_host():
    assert url_features("not a url")["ratio_digits_host"] == 0.0


def test_extract_keeps_index_and_columns():
    urls = pd.Series(["http://a.com", "https://b.org/x"], index=[10, 20])
    df = extract(urls)
    assert list(df.columns) == list(FEATURE_NAMES)
    assert list(df.index) == [10, 20]
