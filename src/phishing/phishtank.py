"""Download and load of the PhishTank list of verified online phishing URLs.

PhishTank only gives phishing URLs, so this file is a test set for recall on
recent phishing, never a training set.
"""

import gzip
import os
import shutil
from pathlib import Path

import pandas as pd
import requests

from phishing import config

BASE_URL = "https://data.phishtank.com/data"
FILE_NAME = "online-valid.csv.gz"
COLUMNS = ["phish_id", "url", "submission_time", "verification_time", "target"]


def download_url(app_key: str | None) -> str:
    """Without a key, PhishTank allows only a few downloads per day."""
    return f"{BASE_URL}/{app_key}/{FILE_NAME}" if app_key else f"{BASE_URL}/{FILE_NAME}"


def download(dest: Path = config.PHISHTANK_FILE, timeout: float = 60.0) -> Path:
    """Download the current list as CSV. Key and User-Agent come from the env."""
    app_key = os.environ.get("PHISHTANK_APP_KEY") or None
    user_agent = os.environ.get("PHISHTANK_USER_AGENT")
    if not user_agent:
        raise RuntimeError("set PHISHTANK_USER_AGENT, for example phishtank/<your-username>")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.gz.part")
    with requests.get(
        download_url(app_key), headers={"User-Agent": user_agent}, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    with gzip.open(tmp, "rb") as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)
    tmp.unlink()
    return dest


def load(path: Path = config.PHISHTANK_FILE) -> pd.DataFrame:
    """Load the list, keep useful columns and add the label (1 = phishing)."""
    df = pd.read_csv(path, usecols=COLUMNS, parse_dates=["submission_time", "verification_time"])
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    df[config.TARGET_COL] = config.LABELS["phishing"]
    return df
