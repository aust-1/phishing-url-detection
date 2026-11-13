"""Lexical features computed from a raw URL string.

Used to build features for new URLs (PhishTank) that are not in the dataset.
Names follow the dataset columns. The definitions are our own version of the
authors' ones: `scripts/check_features.py` compares them with the dataset
values, and only the features that match are used on new URLs.
"""

import ipaddress
from urllib.parse import urlsplit

import pandas as pd

CHAR_COUNTS = {
    "nb_dots": ".",
    "nb_hyphens": "-",
    "nb_at": "@",
    "nb_qm": "?",
    "nb_and": "&",
    "nb_or": "|",
    "nb_eq": "=",
    "nb_underscore": "_",
    "nb_tilde": "~",
    "nb_percent": "%",
    "nb_slash": "/",
    "nb_star": "*",
    "nb_colon": ":",
    "nb_comma": ",",
    "nb_semicolumn": ";",
    "nb_dollar": "$",
    "nb_space": " ",
}

FEATURE_NAMES = (
    "length_url",
    "length_hostname",
    "ip",
    *CHAR_COUNTS,
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _digit_ratio(text: str) -> float:
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


def _port(url: str) -> int:
    try:
        return int(urlsplit(url).port is not None)
    except ValueError:  # port out of range or not a number
        return 1


def url_features(url: str) -> dict[str, float]:
    """Compute the lexical features of one URL."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    lower = url.lower()

    features: dict[str, float] = {
        "length_url": len(url),
        "length_hostname": len(host),
        "ip": int(_is_ip(host)),
    }
    features.update({name: url.count(char) for name, char in CHAR_COUNTS.items()})
    features.update(
        {
            "nb_www": lower.count("www"),
            "nb_com": lower.count("com"),
            # a "//" found after the one of the scheme (redirection)
            "nb_dslash": int(url.rfind("//") > len("https:")),
            "http_in_path": parts.path.lower().count("http"),
            "https_token": int(parts.scheme.lower() != "https"),
            "ratio_digits_url": _digit_ratio(url),
            "ratio_digits_host": _digit_ratio(host),
            "punycode": int("xn--" in lower),
            "port": _port(url),
        }
    )
    return features


def extract(urls: pd.Series) -> pd.DataFrame:
    """Compute the lexical features of a series of URLs."""
    rows = [url_features(str(u)) for u in urls]
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES), index=urls.index)
