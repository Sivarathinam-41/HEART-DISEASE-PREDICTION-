"""
Optional Dataset Downloader
This script can be used to download the alternative expanded 1190-sample
Statlog-Cleveland-Hungary combined dataset from Kaggle.

The active project model uses the local benchmark 'heart.csv' (303 UCI Cleveland records).
"""

import sys

try:
    import kagglehub
    print("[*] Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("sid321axn/heart-statlog-cleveland-hungary-final")
    print("[OK] Dataset downloaded successfully!")
    print(f"Path to dataset files: {path}")
except ImportError:
    print("[INFO] 'kagglehub' is not installed.")
    print("To download the optional 1,190-sample dataset, install it via: pip install kagglehub")
    print("Note: The project already contains the active benchmark dataset 'heart.csv'.")
    sys.exit(0)
