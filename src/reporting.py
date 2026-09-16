"""Stage 5 EDA figures, Stage 11 confusion matrix, Stage 18 application reporting."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.config import FIG_DIR, TEXT_COL, TARGET_COL, CLEAN_COL, SAFE_LABEL, PHISHING_LABEL

sns.set_theme(style="whitegrid")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {path}")
    return path


# ------------------------------- STAGE 5: EDA -------------------------------
def run_eda(df):
    print("\n" + "=" * 60)
    print("STAGE 5 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = df.copy()
    df["text_length"] = df[TEXT_COL].astype(str).str.len()

    print("\nEmail text length statistics (characters):")
    overall = df["text_length"].describe()[["mean", "50%", "min", "max"]]
    print(f"  Mean  : {overall['mean']:.1f}")
    print(f"  Median: {overall['50%']:.1f}")
    print(f"  Min   : {overall['min']:.0f}")
    print(f"  Max   : {overall['max']:.0f}")

    print("\nLength by class:")
    grouped = df.groupby(TARGET_COL)["text_length"].agg(["mean", "median", "min", "max", "count"])
    print(grouped.round(1))

    # Figure 1 — class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    order = [SAFE_LABEL, PHISHING_LABEL]
    sns.countplot(data=df, x=TARGET_COL, order=order, hue=TARGET_COL,
                  palette=["#2a9d8f", "#e76f51"], legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c)
    ax.set_title("Distribution of Safe vs Phishing Emails")
    ax.set_xlabel("Email Type"); ax.set_ylabel("Number of Emails")
    _save(fig, "figure1_class_distribution.png")

    # Figure 2 — length distribution (clipped for readability)
    cap = df["text_length"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df.loc[df["text_length"] <= cap, "text_length"], bins=60, color="#264653", ax=ax)
    ax.set_title(f"Distribution of Email Text Length (clipped at 99th pct = {cap:.0f} chars)")
    ax.set_xlabel("Characters"); ax.set_ylabel("Frequency")
    _save(fig, "figure2_length_distribution.png")

    # Figure 3 — length by class
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df[df["text_length"] <= cap], x=TARGET_COL, y="text_length",
                order=order, hue=TARGET_COL, palette=["#2a9d8f", "#e76f51"],
                legend=False, showfliers=False, ax=ax)
    ax.set_title("Comparison of Email Text Length by Email Type")
    ax.set_xlabel("Email Type"); ax.set_ylabel("Characters")
    _save(fig, "figure3_length_by_class.png")

    # Optional — top words per class
    top_words = {}
    for label in order:
        subset = df.loc[df[TARGET_COL] == label, CLEAN_COL]
        cv = CountVectorizer(stop_words="english", max_features=15)
        counts = cv.fit_transform(subset).sum(axis=0).A1
        pairs = sorted(zip(cv.get_feature_names_out(), counts), key=lambda x: -x[1])
        top_words[label] = pairs
        print(f"\nMost frequent words — {label}:")
        for w, n in pairs[:10]:
            print(f"  {w:<15} {n}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (label, pairs), colour in zip(axes, top_words.items(), ["#2a9d8f", "#e76f51"]):
        words, counts = zip(*pairs[:10])
        ax.barh(list(words)[::-1], list(counts)[::-1], color=colour)
        ax.set_title(f"Top 10 terms — {label}")
        ax.set_xlabel("Frequency")
    fig.suptitle("Word Frequency by Email Class")
    _save(fig, "figure3b_word_frequency.png")

    return {"length_stats": grouped.to_dict(), "top_words": top_words}


# ----------------------- STAGE 11: CONFUSION MATRIX FIG ---------------------
def plot_confusion_matrix(cm, filename="figure7_confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[SAFE_LABEL, PHISHING_LABEL],
                yticklabels=[SAFE_LABEL, PHISHING_LABEL], ax=ax)
    ax.set_xlabel("Predicted Class"); ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix (Test Set)")
    return _save(fig, filename)


def plot_metrics_bar(metrics, filename="figure8_metrics.png"):
    keys = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, [metrics[k] for k in keys], color="#264653")
    ax.bar_label(bars, fmt="%.4f")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Metrics")
    return _save(fig, filename)


# --------------------- STAGE 18: APPLICATION-LEVEL REPORT -------------------
def print_application_report(db_stats):
    total    = db_stats["total"]
    phishing = db_stats["phishing"]
    safe     = db_stats["safe"]
    rate     = (phishing / total * 100) if total else 0.0

    print("\n" + "=" * 45)
    print("        APPLICATION ANALYSIS REPORT")
    print("=" * 45)
    print(f"Total analyses performed : {total}")
    print(f"Phishing detections      : {phishing}")
    print(f"Safe detections          : {safe}")
    print(f"Phishing detection rate  : {rate:.2f}%")
    print(f"First analysis           : {db_stats['first'] or 'n/a'}")
    print(f"Latest analysis          : {db_stats['last'] or 'n/a'}")
    print("-" * 45)
    print("NOTE: this operational detection rate reflects the emails")
    print("submitted to the application. It is NOT the model's test-set")
    print("precision or recall, which are reported separately.")
    print("=" * 45)