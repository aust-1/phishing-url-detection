"""First exploratory analysis on the train split only (no look at val / test).

Usage: python scripts/eda.py   (after make_splits.py)
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from phishing import config
from phishing.data import load_split

HIST_FEATURES = ("length_url", "nb_dots", "domain_age", "page_rank")


def main() -> None:
    train = load_split("train")
    groups = json.loads((config.PROCESSED_DIR / "feature_groups.json").read_text())
    features = [c for cols in groups.values() for c in cols]
    y = train[config.TARGET_COL]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"train rows: {len(train)}, phishing rate {y.mean():.3f}")
    print({name: len(cols) for name, cols in groups.items()})

    constant = [c for c in features if train[c].nunique() == 1]
    print(f"constant features ({len(constant)}): {constant}")

    train[features].describe().T.to_csv(config.REPORTS_DIR / "describe_train.csv")

    corr = train[features].corrwith(y).abs().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    corr.head(15).iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("|Pearson correlation| with target")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "top_correlations.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, name in zip(axes.ravel(), HIST_FEATURES, strict=True):
        for label, value in config.LABELS.items():
            ax.hist(train.loc[y == value, name], bins=40, alpha=0.5, label=label)
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "histograms.png", dpi=150)
    plt.close(fig)

    pd.Series(constant, name="feature").to_csv(
        config.REPORTS_DIR / "constant_features.csv", index=False
    )


if __name__ == "__main__":
    main()
