"""Stages 3-7: loading, quality inspection, cleaning, text preprocessing, encoding."""
import re
import pandas as pd
from src.config import (DATASET_PATH, TEXT_COL, TARGET_COL, CLEAN_COL, INDEX_COL,
                        LABEL_MAP, SAFE_LABEL, PHISHING_LABEL)


# ----------------------------- STAGE 3: LOADING -----------------------------
def load_dataset(path=DATASET_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place Phishing_Email.csv in data/.")
    return pd.read_csv(path)


def inspect_dataset(df):
    """Programmatically verify dataset characteristics — nothing hard-coded."""
    empty_mask = df[TEXT_COL].isna() | (df[TEXT_COL].astype(str).str.strip() == "")
    dup_mask   = df.duplicated(subset=[TEXT_COL, TARGET_COL], keep="first")

    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isna().sum().to_dict(),
        "class_counts": df[TARGET_COL].value_counts().to_dict(),
        "class_pct": (df[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict(),
        "empty_text": int(empty_mask.sum()),
        "duplicates": int(dup_mask.sum()),
    }

    print("=" * 60)
    print("STAGE 3 — DATASET OVERVIEW")
    print("=" * 60)
    print(f"Dataset shape            : {stats['shape']}")
    print(f"Columns                  : {stats['columns']}")
    print("\nData types:")
    for c, t in stats["dtypes"].items():
        print(f"  {c:<15} {t}")
    print("\nFirst five records:")
    print(df.head())
    print("\nMissing values per column:")
    for c, n in stats["missing"].items():
        print(f"  {c:<15} {n}")
    print("\nClass frequencies:")
    for c, n in stats["class_counts"].items():
        print(f"  {c:<15} {n:>6}  ({stats['class_pct'][c]}%)")
    print(f"\nNumber of safe emails    : {stats['class_counts'].get(SAFE_LABEL, 0)}")
    print(f"Number of phishing emails: {stats['class_counts'].get(PHISHING_LABEL, 0)}")
    print(f"Empty email-text records : {stats['empty_text']}")
    print(f"Duplicate records        : {stats['duplicates']}")
    print("=" * 60)
    return stats


# ----------------------------- STAGE 4: CLEANING ----------------------------
def clean_dataset(df):
    print("\n" + "=" * 60)
    print("STAGE 4 — DATA CLEANING")
    print("=" * 60)
    before_rows  = len(df)
    before_dist  = df[TARGET_COL].value_counts().to_dict()
    print(f"Rows before cleaning     : {before_rows}")
    print(f"Class distribution before: {before_dist}")

    # 1. drop index column
    if INDEX_COL in df.columns:
        df = df.drop(columns=[INDEX_COL])
        print(f"Dropped column '{INDEX_COL}' (row index, not predictive).")

    # 2-4. remove null / empty / whitespace-only text
    empty_mask = df[TEXT_COL].isna() | (df[TEXT_COL].astype(str).str.strip() == "")
    n_empty = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()
    print(f"Removed empty/whitespace records : {n_empty} (no text to classify)")

    # 5-6. remove exact duplicates
    dup_mask = df.duplicated(subset=[TEXT_COL, TARGET_COL], keep="first")
    n_dup = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()
    print(f"Removed exact duplicate records  : {n_dup} (prevents train/test leakage)")

    # 7. validate target labels
    labels = set(df[TARGET_COL].unique())
    unexpected = labels - {SAFE_LABEL, PHISHING_LABEL}
    if unexpected:
        raise ValueError(f"Unexpected target labels found: {unexpected}")
    print(f"Target labels verified           : {sorted(labels)}")

    df = df.reset_index(drop=True)
    print(f"\nRows after cleaning      : {len(df)}")
    print(f"Class distribution after : {df[TARGET_COL].value_counts().to_dict()}")
    print("=" * 60)

    summary = {"rows_before": before_rows, "rows_after": len(df),
               "removed_empty": n_empty, "removed_duplicates": n_dup}
    return df, summary


# ------------------------ STAGE 6: TEXT PREPROCESSING -----------------------
HTML_TAG_RE   = re.compile(r"<[^>]+>")
URL_RE        = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE      = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text):
    """Conservative cleaning that PRESERVES phishing indicators.

    URLs and email addresses are replaced with tokens rather than deleted, so
    the presence of a link remains a usable feature. Punctuation, currency
    symbols and urgency wording are retained.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = HTML_TAG_RE.sub(" ", t)            # strip HTML markup
    t = URL_RE.sub(" urltoken ", t)        # keep link presence as a signal
    t = EMAIL_RE.sub(" emailtoken ", t)    # keep sender/address presence
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = WHITESPACE_RE.sub(" ", t)          # normalise whitespace
    return t.strip()


def add_cleaned_column(df, n_examples=3):
    df[CLEAN_COL] = df[TEXT_COL].apply(preprocess_text)
    df = df[df[CLEAN_COL].str.strip() != ""].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("STAGE 6 — PREPROCESSING VALIDATION EXAMPLES")
    print("=" * 60)
    for i in range(min(n_examples, len(df))):
        print(f"\n--- Example {i+1} ({df.loc[i, TARGET_COL]}) ---")
        print(f"ORIGINAL: {str(df.loc[i, TEXT_COL])[:220]}")
        print(f"CLEANED : {df.loc[i, CLEAN_COL][:220]}")
    print("=" * 60)
    return df


# -------------------------- STAGE 7: TARGET ENCODING ------------------------
def encode_target(df):
    y = df[TARGET_COL].map(LABEL_MAP)
    if y.isna().any():
        raise ValueError("Unmapped target values present after encoding.")
    X = df[CLEAN_COL]
    print("\nSTAGE 7 — TARGET ENCODING")
    print(f"  Mapping    : {LABEL_MAP}")
    print(f"  y counts   : {y.value_counts().to_dict()}  (0=Safe, 1=Phishing)")
    print(f"  X samples  : {len(X)}")
    return X, y