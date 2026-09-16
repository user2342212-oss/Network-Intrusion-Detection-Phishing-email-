"""Entry point. Usage:
    python main.py train    -> run stages 3-14 and save artefacts
    python main.py cli      -> run the command-line application
    python main.py test     -> run the functional test plan
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ALL_DIRS
from src.data_preprocessing import (load_dataset, inspect_dataset, clean_dataset,
                                    add_cleaned_column, encode_target)
from src.reporting import run_eda
from src.feature_extraction import fit_transform_features
from src.model_training import (split_data, train_logistic_regression, evaluate_model,
                                compare_with_random_forest, error_analysis,
                                save_artefacts, verify_artefacts)
from src.database import initialise_database


def train_pipeline(run_rf=True):
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    initialise_database()

    df = load_dataset()                       # Stage 3
    inspect_dataset(df)
    df, clean_summary = clean_dataset(df)     # Stage 4
    df = add_cleaned_column(df)               # Stage 6
    run_eda(df)                               # Stage 5
    X, y = encode_target(df)                  # Stage 7

    X_train, X_test, y_train, y_test = split_data(X, y)          # Stage 8
    vec, Xtr, Xte = fit_transform_features(X_train, X_test)      # Stage 9
    clf = train_logistic_regression(Xtr, y_train)                # Stage 10
    metrics, y_pred = evaluate_model(clf, Xte, y_test)           # Stage 11
    if run_rf:
        compare_with_random_forest(Xtr, y_train, Xte, y_test)    # Stage 12
    error_analysis(X_test, y_test, y_pred)                       # Stage 13
    save_artefacts(vec, clf, metrics)                            # Stage 14
    verify_artefacts()
    print("\nTraining pipeline complete. Run: python main.py cli")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if cmd == "train":
        train_pipeline()
    elif cmd == "cli":
        from src.cli import run_cli
        run_cli()
    elif cmd == "test":
        from tests.test_system import run_tests
        run_tests()
    else:
        print("Usage: python main.py [train|cli|test]")
