"""Stages 15-16: SQLite integration with parameterised queries only."""
import sqlite3
from datetime import datetime
from src.config import DB_PATH, DB_DIR, MAX_STORED_TEXT_CHARS, PREPROCESSING_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis (
    analysis_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    email_text            TEXT    NOT NULL,
    prediction            TEXT    NOT NULL,
    prediction_score      REAL    NOT NULL,
    model_name            TEXT,
    text_length           INTEGER,
    top_features          TEXT,
    preprocessing_version TEXT,
    analysis_timestamp    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prediction ON analysis(prediction);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON analysis(analysis_timestamp);
"""


def _connect():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_database():
    with _connect() as conn:
        conn.executescript(SCHEMA)
    print(f"  Database ready at {DB_PATH}")
    return True


def save_analysis(email_text, prediction, prediction_score,
                  model_name=None, top_features=None):
    """Stores a truncated copy of the email only — full bodies are not retained."""
    stored = (email_text or "")[:MAX_STORED_TEXT_CHARS]
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO analysis
               (email_text, prediction, prediction_score, model_name,
                text_length, top_features, preprocessing_version, analysis_timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (stored, prediction, float(prediction_score), model_name,
             len(email_text or ""), top_features, PREPROCESSING_VERSION,
             datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid


def get_analysis_history(limit=10):
    with _connect() as conn:
        return conn.execute(
            """SELECT analysis_id, prediction, prediction_score, text_length,
                      analysis_timestamp, substr(email_text, 1, 60) AS preview
               FROM analysis ORDER BY analysis_id DESC LIMIT ?""", (limit,)).fetchall()


def get_phishing_attempts(limit=10):
    with _connect() as conn:
        return conn.execute(
            """SELECT analysis_id, prediction_score, text_length,
                      analysis_timestamp, substr(email_text, 1, 60) AS preview
               FROM analysis WHERE prediction = ?
               ORDER BY analysis_id DESC LIMIT ?""", ("Phishing Email", limit)).fetchall()


def get_analysis_by_id(analysis_id):
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM analysis WHERE analysis_id = ?", (analysis_id,)).fetchone()


def get_summary_statistics():
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN prediction = ? THEN 1 ELSE 0 END) AS phishing,
                      SUM(CASE WHEN prediction = ? THEN 1 ELSE 0 END) AS safe,
                      MIN(analysis_timestamp) AS first,
                      MAX(analysis_timestamp) AS last
               FROM analysis""", ("Phishing Email", "Safe Email")).fetchone()
    return {"total": row["total"] or 0, "phishing": row["phishing"] or 0,
            "safe": row["safe"] or 0, "first": row["first"], "last": row["last"]}