"""Download the PhishTank list and compute its lexical features.

Usage:
    export PHISHTANK_USER_AGENT=phishtank/<your-username>
    export PHISHTANK_APP_KEY=<key>   # optional, removes the daily limit
    python scripts/fetch_phishtank.py
"""

from phishing import config, phishtank
from phishing.features import extract


def main() -> None:
    path = phishtank.download()
    df = phishtank.load(path)
    features = extract(df[config.URL_COL])
    out = df[[config.URL_COL, "submission_time", config.TARGET_COL]].join(features)

    out_path = config.PROCESSED_DIR / "phishtank_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"{len(out)} URLs, submitted from {df.submission_time.min()} to "
        f"{df.submission_time.max()}, saved to {out_path}"
    )


if __name__ == "__main__":
    main()
