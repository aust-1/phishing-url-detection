"""Clean the raw dataset and write the train / val / test splits.

Usage: python scripts/make_splits.py
"""

import json

from phishing import config
from phishing.data import clean, feature_groups, load_raw, save_splits, split


def main() -> None:
    raw = load_raw()
    df = clean(raw)
    print(f"raw rows: {len(raw)}, after cleaning: {len(df)}")

    groups = feature_groups(list(df.columns))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "feature_groups.json").write_text(json.dumps(groups, indent=2) + "\n")

    splits = split(df)
    save_splits(splits)
    for name, part in splits.items():
        rate = part[config.TARGET_COL].mean()
        print(f"{name:>5}: {len(part):>5} rows, phishing rate {rate:.3f}")


if __name__ == "__main__":
    main()
