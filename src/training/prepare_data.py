# src/training/prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


RAW_PATH = Path("data/urls.csv")
OUT_DIR = Path("data/processed")

URL_COL = "url"
LABEL_COL = "is_spam"   # True/False


def main():
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning
    df = df.dropna(subset=[URL_COL, LABEL_COL]).copy()
    df[URL_COL] = df[URL_COL].astype(str).str.strip()
    df = df[df[URL_COL] != ""]

    # Normalize labels: True/False -> 1/0
    # Handles booleans, "True"/"False" strings, etc.
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().map({"true": 1, "false": 0})
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Remove duplicates
    df = df.drop_duplicates(subset=[URL_COL])

    # Stratified split: train/val/test = 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df[LABEL_COL]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df[LABEL_COL]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print("Saved:")
    print(" -", OUT_DIR / "train.csv", len(train_df))
    print(" -", OUT_DIR / "val.csv", len(val_df))
    print(" -", OUT_DIR / "test.csv", len(test_df))
    print("\nLabel distribution (train):")
    print(train_df[LABEL_COL].value_counts(normalize=True))


if __name__ == "__main__":
    main()
