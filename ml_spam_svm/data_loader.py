import os
import requests
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "sms_spam_no_header.csv")


def ensure_data(download=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        return DATA_FILE
    if not download:
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    print("Downloading dataset...")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    with open(DATA_FILE, 'wb') as f:
        f.write(r.content)
    return DATA_FILE


def load_sms_spam(download=True):
    path = ensure_data(download=download)
    # CSV has no header, two columns: label,text
    df = pd.read_csv(path, header=None, names=['label', 'text'], encoding='utf-8')
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df
