
"""Stages 17 & 19: command-line interface with input validation and safe handling."""
import sys
from src.config import INVERSE_MAP, MAX_STORED_TEXT_CHARS
from src.data_preprocessing import preprocess_text
from src.model_training import load_artefacts
from src import database as db
from src.reporting import print_application_report

MAX_INPUT_CHARS = 100_000
BANNER = """
=========================================
      PHISHING EMAIL DETECTOR
=========================================
1. Analyse Email
2. View Analysis History
3. View Phishing Attempts
4. View Model Information
5. View Reporting Summary
6. Exit
"""


class Detector:
    """Wraps the saved artefacts. Email text is treated strictly as untrusted
    data: it is never evaluated, executed, opened or requested over a network."""

    def __init__(self):
        self.vectorizer, self.classifier, self.metadata = load_artefacts()
        self.feature_names = self.vectorizer.get_feature_names_out()

    def predict(self, raw_text):
        cleaned = preprocess_text(raw_text)
        vector  = self.vectorizer.transform([cleaned])
        label   = int(self.classifier.predict(vector)[0])
        score   = float(self.classifier.predict_proba(vector)[0][1])
        return INVERSE_MAP[label], score, self._top_features(vector)

    def _top_features(self, vector, k=5):
        """Highest TF-IDF terms present in the submitted email — stored as a
        feature summary rather than retaining the full message."""
        arr = vector.toarray().ravel()
        idx = arr.argsort()[-k:][::-1]
        return ", ".join(self.feature_names[i] for i in idx if arr[i] > 0)


def _validate_email_input(text):
    if text is None or not text.strip():
        return False, "Email text cannot be empty."
    if len(text) > MAX_INPUT_CHARS:
        return False, f"Input exceeds {MAX_INPUT_CHARS} characters and was rejected."
    return True, ""


def _read_multiline():
    print("Enter email text (finish with a line containing only END):")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def analyse_email(detector):
    raw = _read_multiline()
    ok, msg = _validate_email_input(raw)
    if not ok:
        print(f"[INPUT ERROR] {msg}")
        return
    print("\nAnalysing...")
    try:
        label, score, feats = detector.predict(raw)
    except Exception as e:                        # safe error handling
        print(f"[ERROR] Analysis failed: {type(e).__name__}")
        return

    print("\nPrediction:")
    print("POTENTIAL PHISHING EMAIL" if label == "Phishing Email" else "SAFE EMAIL")
    print(f"Phishing probability : {score:.4f}")
    print(f"Top contributing terms: {feats or 'n/a'}")

    try:
        rid = db.save_analysis(raw, label, score,
                               model_name=detector.metadata.get("model_name"),
                               top_features=feats)
        print(f"Analysis saved successfully (analysis_id={rid}).")
        print(f"Note: only the first {MAX_STORED_TEXT_CHARS} characters are retained.")
    except Exception as e:
        print(f"[ERROR] Could not save analysis: {type(e).__name__}")


def view_history():
    rows = db.get_analysis_history()
    if not rows:
        print("No analyses recorded yet.")
        return
    print(f"\n{'ID':<5}{'Prediction':<17}{'Score':<9}{'Len':<8}{'Timestamp':<21}Preview")
    print("-" * 100)
    for r in rows:
        print(f"{r['analysis_id']:<5}{r['prediction']:<17}{r['prediction_score']:<9.4f}"
              f"{r['text_length']:<8}{r['analysis_timestamp']:<21}{(r['preview'] or '')[:35]}")


def view_phishing():
    rows = db.get_phishing_attempts()
    if not rows:
        print("No phishing attempts recorded yet.")
        return
    print(f"\n{'ID':<5}{'Score':<9}{'Len':<8}{'Timestamp':<21}Preview")
    print("-" * 90)
    for r in rows:
        print(f"{r['analysis_id']:<5}{r['prediction_score']:<9.4f}{r['text_length']:<8}"
              f"{r['analysis_timestamp']:<21}{(r['preview'] or '')[:35]}")


def view_model_info(detector):
    m = detector.metadata
    print("\n--- MODEL INFORMATION ---")
    print(f"Model                 : {m.get('model_name', 'n/a')}")
    print(f"Trained at            : {m.get('trained_at', 'n/a')}")
    print(f"TF-IDF features       : {m.get('n_features', 'n/a')}")
    print(f"Preprocessing version : {m.get('preprocessing_version', 'n/a')}")
    print(f"Random state          : {m.get('random_state', 'n/a')}")
    for k, v in (m.get("metrics") or {}).items():
        print(f"  test {k:<10}: {v:.4f}")


def run_cli():
    db.initialise_database()
    try:
        detector = Detector()
    except FileNotFoundError as e:
        print(f"[STARTUP ERROR] {e}")
        sys.exit(1)

    actions = {
        "1": lambda: analyse_email(detector),
        "2": view_history,
        "3": view_phishing,
        "4": lambda: view_model_info(detector),
        "5": lambda: print_application_report(db.get_summary_statistics()),
    }
    while True:
        print(BANNER)
        choice = input("Select an option: ").strip()
        if choice == "6":
            print("Exiting. Goodbye.")
            break
        action = actions.get(choice)
        if action is None:
            print("[INVALID OPTION] Please choose a number between 1 and 6.")
            continue
        action()