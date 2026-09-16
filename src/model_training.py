"""Stages 8, 10-14: split, training, evaluation, robustness, error analysis, artefacts."""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from src.config import (RANDOM_STATE, TEST_SIZE, VECTORIZER_PATH, CLASSIFIER_PATH,
                        METADATA_PATH, MODEL_DIR, MODEL_NAME, PREPROCESSING_VERSION,
                        INVERSE_MAP)
from src.feature_extraction import fit_transform_features
from src.reporting import plot_confusion_matrix, plot_metrics_bar


# ------------------------- STAGE 8: TRAIN/TEST SPLIT ------------------------
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("\n" + "=" * 60)
    print("STAGE 8 — TRAIN/TEST SPLIT")
    print("=" * 60)
    print(f"Train size: {len(X_train)}  class balance: "
          f"{np.round(y_train.value_counts(normalize=True).sort_index().values, 4)}")
    print(f"Test size : {len(X_test)}  class balance: "
          f"{np.round(y_test.value_counts(normalize=True).sort_index().values, 4)}")
    print(f"Stratified on target, random_state={RANDOM_STATE}")
    print("=" * 60)
    return X_train, X_test, y_train, y_test


# --------------------------- STAGE 10: TRAINING -----------------------------
def train_logistic_regression(Xtr, y_train):
    print("\nSTAGE 10 — TRAINING LOGISTIC REGRESSION")
    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver="liblinear",
        class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(Xtr, y_train)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Solver     : liblinear | class_weight=balanced | C=1.0")
    print(f"  Trained on : {Xtr.shape[0]} samples x {Xtr.shape[1]} features")
    return clf


# -------------------------- STAGE 11: EVALUATION ----------------------------
def evaluate_model(clf, Xte, y_test, make_figures=True, tag=""):
    y_pred = clf.predict(Xte)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall":    recall_score(y_test, y_pred, pos_label=1),
        "f1":        f1_score(y_test, y_pred, pos_label=1),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print(f"STAGE 11 — MODEL EVALUATION (unseen test set) {tag}")
    print("=" * 60)
    print(f"{'Metric':<12}| Result")
    print("-" * 28)
    for k, v in metrics.items():
        print(f"{k.capitalize():<12}| {v:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"                 Pred Safe   Pred Phishing")
    print(f"  Actual Safe    {tn:>9}   {fp:>13}")
    print(f"  Actual Phish   {fn:>9}   {tp:>13}")
    print(f"\n  True Negatives  (TN) = {tn}  safe email correctly allowed")
    print(f"  False Positives (FP) = {fp}  safe email wrongly flagged -> usability/trust cost")
    print(f"  False Negatives (FN) = {fn}  phishing missed -> direct security risk")
    print(f"  True Positives  (TP) = {tp}  phishing correctly detected")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Safe Email", "Phishing Email"], digits=4))
    print("=" * 60)

    if make_figures:
        plot_confusion_matrix(cm)
        plot_metrics_bar(metrics)

    metrics["confusion_matrix"] = cm.tolist()
    return metrics, y_pred


# ------------------- STAGE 12: OPTIONAL ROBUSTNESS COMPARISON ---------------
def compare_with_random_forest(Xtr, y_train, Xte, y_test):
    print("\n" + "=" * 60)
    print("STAGE 12 — ROBUSTNESS COMPARISON: RANDOM FOREST")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced_subsample",
        random_state=RANDOM_STATE)
    rf.fit(Xtr, y_train)
    rf_metrics, _ = evaluate_model(rf, Xte, y_test, make_figures=False, tag="[Random Forest]")
    print("Note: identical split and metrics used. Logistic Regression remains the")
    print("primary model — it is linear in a sparse high-dimensional space, trains")
    print("in seconds, and its coefficients are directly interpretable, whereas the")
    print("forest is far more expensive and harder to explain to an analyst.")
    return rf_metrics


# ------------------------- STAGE 13: ERROR ANALYSIS -------------------------
def error_analysis(X_test_raw, y_test, y_pred, n=3, snippet=180):
    print("\n" + "=" * 60)
    print("STAGE 13 — ERROR ANALYSIS")
    print("=" * 60)
    df = pd.DataFrame({"text": X_test_raw.values,
                       "actual": y_test.values, "pred": y_pred})
    fps = df[(df.actual == 0) & (df.pred == 1)].head(n)
    fns = df[(df.actual == 1) & (df.pred == 0)].head(n)

    for title, frame in [("FALSE POSITIVES (safe flagged as phishing)", fps),
                         ("FALSE NEGATIVES (phishing classified as safe)", fns)]:
        print(f"\n--- {title} — showing {len(frame)} example(s) ---")
        for i, row in frame.iterrows():
            print(f"  Actual: {INVERSE_MAP[row.actual]} | Predicted: {INVERSE_MAP[row.pred]}")
            print(f"  Text  : {str(row.text)[:snippet]} ...\n")
    print("Likely causes: FPs occur in legitimate marketing/IT-notification emails")
    print("that share vocabulary with phishing (account, verify, click). FNs occur")
    print("in very short messages where the model has too few tokens to score.")
    print("=" * 60)
    return fps, fns


# ---------------------- STAGE 14: SAVE & VERIFY ARTEFACTS -------------------
def save_artefacts(vectorizer, classifier, metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    metadata = {
        "model_name": MODEL_NAME,
        "preprocessing_version": PREPROCESSING_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "n_features": len(vectorizer.get_feature_names_out()),
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    }
    joblib.dump(metadata, METADATA_PATH)
    print("\nSTAGE 14 — ARTEFACTS SAVED")
    for p in (VECTORIZER_PATH, CLASSIFIER_PATH, METADATA_PATH):
        print(f"  {p}  exists={p.exists()}")
    return metadata


def load_artefacts():
    for p in (VECTORIZER_PATH, CLASSIFIER_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing artefact {p}. Run: python main.py train")
    return (joblib.load(VECTORIZER_PATH),
            joblib.load(CLASSIFIER_PATH),
            joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {})


def verify_artefacts(sample_text="Dear user, verify your account now at http://secure-login.example"):
    from src.data_preprocessing import preprocess_text
    vec, clf, meta = load_artefacts()
    vector = vec.transform([preprocess_text(sample_text)])
    pred = int(clf.predict(vector)[0])
    prob = float(clf.predict_proba(vector)[0][1])
    print(f"  Reload test -> prediction={INVERSE_MAP[pred]}  phishing_probability={prob:.4f}")
    return True
