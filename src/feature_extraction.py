"""Stage 9: TF-IDF feature extraction, fitted on training data only."""
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import RANDOM_STATE


def build_vectorizer(max_features=50000, ngram_range=(1, 2),
                     min_df=3, max_df=0.90, sublinear_tf=True):
    """Parameter justification:
    - ngram_range=(1,2): bigrams capture phishing phrases ('click here',
      'verify account') that unigrams alone lose.
    - min_df=3: a term must appear in >=3 emails; removes typos/hashes that
      would otherwise inflate dimensionality and overfit.
    - max_df=0.90: drops terms present in >90% of emails (boilerplate) which
      carry no discriminative power.
    - sublinear_tf=True: 1+log(tf) damps very long emails where raw counts
      would dominate; email lengths are heavily right-skewed (see Figure 2).
    - max_features caps the matrix at a tractable size for Logistic Regression.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        stop_words="english",
        lowercase=True,
    )


def fit_transform_features(X_train, X_test, **kwargs):
    vec = build_vectorizer(**kwargs)
    Xtr = vec.fit_transform(X_train)    # FIT ONLY ON TRAIN — no leakage
    Xte = vec.transform(X_test)         # test set is transformed only

    print("\n" + "=" * 60)
    print("STAGE 9 — TF-IDF FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Training samples         : {Xtr.shape[0]}")
    print(f"Testing samples          : {Xte.shape[0]}")
    print(f"TF-IDF features generated: {Xtr.shape[1]}")
    print(f"Training matrix sparsity : {100*(1 - Xtr.nnz/(Xtr.shape[0]*Xtr.shape[1])):.3f}% zeros")
    print(f"Parameters               : ngram={vec.ngram_range}, min_df={vec.min_df}, "
          f"max_df={vec.max_df}, sublinear_tf={vec.sublinear_tf}")
    print("=" * 60)
    return vec, Xtr, Xte