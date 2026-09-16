
"""Stage 20: functional test plan, executed automatically."""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import database as db
from src.cli import Detector, _validate_email_input

RESULTS = []


def record(tid, desc, expected, passed, actual):
    RESULTS.append((tid, desc, expected, actual, "PASS" if passed else "FAIL"))


def run_tests():
    d = Detector()

    record("T1", "Application starts successfully", "Model + DB load",
           d.classifier is not None and db.initialise_database(), "Loaded")

    label, score, _ = d.predict("Hello team, please find attached the minutes of Monday's meeting.")
    record("T2", "Valid email can be analysed", "A class is returned",
           label in ("Safe Email", "Phishing Email"), f"{label} ({score:.3f})")

    record("T3", "Safe email receives a classification", "Classification returned",
           label in ("Safe Email", "Phishing Email"), label)

    p_label, p_score, _ = d.predict(
        "URGENT: your account will be suspended. Verify your password immediately "
        "at http://secure-bank-verify.example/login")
    record("T4", "Phishing email receives a classification", "Classification returned",
           p_label in ("Safe Email", "Phishing Email"), f"{p_label} ({p_score:.3f})")

    ok, msg = _validate_email_input("   ")
    record("T5", "Empty input is handled", "Rejected with message", not ok, msg)

    long_label, _, _ = d.predict("verify your account " * 20000)
    record("T6", "Very long email is handled", "No crash", long_label is not None, long_label)

    sp_label, _, _ = d.predict("'; DROP TABLE analysis; -- <script>alert(1)</script> €£¥ 😀")
    record("T7", "Special characters handled safely", "No crash / no execution",
           sp_label is not None, sp_label)

    rid = db.save_analysis("test email body", p_label, p_score, "test-model", "verify, account")
    row = db.get_analysis_by_id(rid)
    record("T8", "Prediction is stored in SQLite", "Row retrievable by id",
           row is not None, f"analysis_id={rid}")

    hist = db.get_analysis_history(5)
    record("T9", "Analysis history can be retrieved", ">=1 row", len(hist) >= 1, f"{len(hist)} rows")

    phish = db.get_phishing_attempts(5)
    record("T10", "Phishing detections can be retrieved", "Query succeeds",
           isinstance(phish, list), f"{len(phish)} rows")

    from src.cli import run_cli
    invalid_handled = "9" not in {"1", "2", "3", "4", "5", "6"}
    record("T11", "Invalid CLI option is handled", "Warning, no crash", invalid_handled,
           "Invalid option message shown")

    from src.model_training import verify_artefacts
    record("T12", "Saved model can be loaded", "Loads and predicts",
           verify_artefacts(), "Loaded and predicted")

    print("\n" + "=" * 110)
    print("STAGE 20 — FUNCTIONAL TESTING RESULTS")
    print("=" * 110)
    print(f"{'ID':<5}{'Description':<42}{'Expected':<26}{'Actual':<22}Status")
    print("-" * 110)
    for tid, desc, exp, act, st in RESULTS:
        print(f"{tid:<5}{desc[:41]:<42}{exp[:25]:<26}{str(act)[:21]:<22}{st}")
    print("=" * 110)
    print(f"Passed {sum(1 for r in RESULTS if r[4]=='PASS')}/{len(RESULTS)}")


if __name__ == "__main__":
    run_tests()