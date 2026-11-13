"""Loading, cleaning and splitting of the Kaggle / Mendeley dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from phishing import config

SPLITS = ("train", "val", "test")


def load_raw(path: Path = config.KAGGLE_FILE) -> pd.DataFrame:
    """Read the raw CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, see data/README.md to download it")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Map the target to 0/1 and remove duplicated URLs.

    A URL seen twice with two different labels is dropped, since we cannot know
    the true label.
    """
    df = df.copy()

    unknown = set(df[config.TARGET_COL].unique()) - set(config.LABELS)
    if unknown:
        raise ValueError(f"unknown labels in {config.TARGET_COL!r}: {sorted(unknown)}")
    df[config.TARGET_COL] = df[config.TARGET_COL].map(config.LABELS).astype("int8")

    n_labels = df.groupby(config.URL_COL)[config.TARGET_COL].transform("nunique")
    df = df[n_labels == 1]
    df = df.drop_duplicates(subset=config.URL_COL, keep="first")

    if df.isna().any().any():
        raise ValueError("missing values found, check the raw file")
    return df.reset_index(drop=True)


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Return the URL, content and external feature columns.

    Groups are found from their first column and the order of the dataset.
    The `url` and target columns are never part of a group.
    """
    features = [c for c in columns if c not in (config.URL_COL, config.TARGET_COL)]
    starts = {name: features.index(first) for name, first in config.GROUP_STARTS.items()}
    order = sorted(starts, key=starts.get)
    bounds = [starts[name] for name in order] + [len(features)]
    if bounds[0] != 0:
        raise ValueError(f"unexpected columns before {config.GROUP_STARTS[order[0]]!r}")

    groups = {name: features[bounds[i] : bounds[i + 1]] for i, name in enumerate(order)}
    for name, cols in groups.items():
        expected = config.GROUP_SIZES[name]
        if len(cols) != expected:
            raise ValueError(f"group {name!r} has {len(cols)} columns, expected {expected}")
    return groups


def split(
    df: pd.DataFrame,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split."""
    y = df[config.TARGET_COL]
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val[config.TARGET_COL],
        random_state=random_state,
    )
    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path = config.PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)


def load_split(name: str, in_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {name!r}")
    return pd.read_csv(in_dir / f"{name}.csv")


def xy(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and the target. The `url` column is never kept."""
    if config.URL_COL in columns:
        raise ValueError("the url column must not be used as a feature")
    return df[columns], df[config.TARGET_COL]
