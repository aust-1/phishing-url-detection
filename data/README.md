# Data

Nothing in this folder is committed, only the folder structure.

| Folder       | Content                                            |
| ------------ | -------------------------------------------------- |
| `raw/`       | `dataset_phishing.csv` from Kaggle, never modified |
| `external/`  | PhishTank list (`phishtank_online_valid.csv`)      |
| `processed/` | splits and files written by the scripts            |

## Kaggle / Mendeley dataset

Web page Phishing Detection Dataset, Hannousse and Yahiouche (2021).
11,430 URLs, 87 features (56 URL, 24 content, 7 external), 50% phishing.

- Kaggle: <https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset>
- Original source: <https://data.mendeley.com/datasets/c2gw7fy2j4/3>

Download with the Kaggle CLI (API token in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d shashwatwork/web-page-phishing-detection-dataset -p data/raw --unzip
```

or download the zip from the Kaggle page and put `dataset_phishing.csv` in `data/raw/`.

## PhishTank

List of verified phishing URLs that are still online, updated every hour.
Only phishing URLs, so it is used as an extra test set, never for training.
See `scripts/fetch_phishtank.py` and <https://phishtank.net/developer_info.php>.
