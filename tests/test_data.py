import numpy as np
import pandas as pd
import pytest

from phishing import config
from phishing.data import clean, feature_groups, split, xy


def _columns() -> list[str]:
    url = ["length_url"] + [f"u{i}" for i in range(55)]
    content = ["nb_hyperlinks"] + [f"c{i}" for i in range(23)]
    external = ["whois_registered_domain"] + [f"e{i}" for i in range(6)]
    return [config.URL_COL, *url, *content, *external, config.TARGET_COL]


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = _columns()
    df = pd.DataFrame(rng.integers(0, 5, size=(n, len(cols))), columns=cols)
    df[config.URL_COL] = [f"http://site{i}.com" for i in range(n)]
    df[config.TARGET_COL] = ["phishing", "legitimate"] * (n // 2)
    return df


def test_clean_maps_target_and_drops_conflicts():
    df = _frame(10)
    dup = df.iloc[[0]].copy()
    conflict = df.iloc[[1]].copy()
    conflict[config.TARGET_COL] = (
        "phishing" if df.at[1, config.TARGET_COL] == "legitimate" else "legitimate"
    )
    out = clean(pd.concat([df, dup, conflict], ignore_index=True))

    assert set(out[config.TARGET_COL].unique()) <= {0, 1}
    assert out[config.URL_COL].is_unique
    assert df.at[1, config.URL_COL] not in set(out[config.URL_COL])
    assert len(out) == 9


def test_clean_rejects_unknown_label():
    df = _frame(4)
    df.loc[0, config.TARGET_COL] = "spam"
    with pytest.raises(ValueError):
        clean(df)


def test_feature_groups_sizes():
    groups = feature_groups(_columns())
    assert {k: len(v) for k, v in groups.items()} == config.GROUP_SIZES
    assert config.URL_COL not in sum(groups.values(), [])


def test_split_sizes_and_stratification():
    df = clean(_frame(1000))
    parts = split(df)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert len(parts["test"]) == 150
    assert len(parts["val"]) == 150
    for p in parts.values():
        assert p[config.TARGET_COL].mean() == pytest.approx(0.5, abs=0.02)
    urls = [set(p[config.URL_COL]) for p in parts.values()]
    assert not (urls[0] & urls[1] or urls[0] & urls[2] or urls[1] & urls[2])


def test_xy_refuses_url_column():
    with pytest.raises(ValueError):
        xy(_frame(4), [config.URL_COL])
