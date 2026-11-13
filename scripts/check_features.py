"""Compare our lexical features with the values of the dataset.

A feature is "portable" (usable on new PhishTank URLs) when our value is the
same as the dataset value for at least --min-match of the rows.

Usage: python scripts/check_features.py [--min-match 0.99]
"""

import argparse
import json

import numpy as np
import pandas as pd

from phishing import config
from phishing.data import clean, load_raw
from phishing.features import FEATURE_NAMES, extract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-match", type=float, default=0.99)
    args = parser.parse_args()

    df = clean(load_raw())
    ours = extract(df[config.URL_COL])

    rows = []
    for name in FEATURE_NAMES:
        if name not in df.columns:
            rows.append({"feature": name, "match_rate": np.nan})
            continue
        same = np.isclose(ours[name].to_numpy(float), df[name].to_numpy(float), atol=1e-3)
        rows.append({"feature": name, "match_rate": same.mean()})

    report = pd.DataFrame(rows).sort_values("match_rate", ascending=False)
    report["portable"] = report["match_rate"] >= args.min_match
    print(report.to_string(index=False))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.REPORTS_DIR / "feature_check.csv", index=False)
    portable = report.loc[report["portable"], "feature"].tolist()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "portable_features.json").write_text(
        json.dumps(portable, indent=2) + "\n"
    )
    print(f"\n{len(portable)} / {len(FEATURE_NAMES)} portable features")


if __name__ == "__main__":
    main()
