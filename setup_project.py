"""Stage 1 & 2: environment setup, directory creation, database initialisation."""
import subprocess, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def install_dependencies():
    req = BASE_DIR / "requirements.txt"
    if not req.exists():
        print("[WARN] requirements.txt not found; skipping install.")
        return
    print("[1/3] Installing dependencies ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def create_directories():
    from src.config import ALL_DIRS
    print("[2/3] Creating project directories ...")
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"      ok -> {d.relative_to(BASE_DIR)}")


def initialise_db():
    from src.database import initialise_database
    print("[3/3] Initialising SQLite database ...")
    initialise_database()


if __name__ == "__main__":
    install_dependencies()
    create_directories()
    initialise_db()
    print("\nSetup complete. Place Phishing_Email.csv in data/ then run: python main.py train")