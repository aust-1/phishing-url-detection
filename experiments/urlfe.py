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


def registered_domain(url: str) -> str:
    try:
        host = urlsplit(str(url)).hostname or str(url)
    except ValueError:
        host = str(url)
    r = _EX(host)
    return r.top_domain_under_public_suffix or r.domain or host
