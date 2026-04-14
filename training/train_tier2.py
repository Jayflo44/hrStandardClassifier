from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "hrStandardClassifier" / "data" / "processed" / "train_clean_validated.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TEXT_COL = "comment_text"
ID_COL = "id"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 3
WORD_MAX_FEATURES = 15000
CHAR_MAX_FEATURES = 15000
SVD_COMPONENTS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df["is_toxic"] = df[LABEL_COLS].max(axis=1).astype(int)
    return df

def build_feature_extractors() -> tuple[FeatureUnion, TruncatedSVD]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=WORD_MAX_FEATURES,
        stop_words="english",
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        max_features=CHAR_MAX_FEATURES,
    )

    combined_features = FeatureUnion([
        ("words", word_vectorizer),
        ("chars", char_vectorizer),
    ])

    dense_projector = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    return combined_features, dense_projector

def extract_features(
    train_texts: pd.Series,
    eval_texts: pd.Series,
    feature_union: FeatureUnion,
    dense_projector: TruncatedSVD,
) -> tuple[np.ndarray, np.ndarray, object, np.ndarray, np.ndarray]:
    X_train_tfidf = feature_union.fit_transform(train_texts)
    X_eval_tfidf = feature_union.transform(eval_texts)

    X_train_dense = dense_projector.fit_transform(X_train_tfidf)
    X_eval_dense = dense_projector.transform(X_eval_tfidf)

    X_train_final = hstack([X_train_tfidf, X_train_dense]).tocsr()
    X_eval_final = hstack([X_eval_tfidf, X_eval_dense]).tocsr()

    return X_train_final, X_eval_final, feature_union, X_train_dense, X_eval_dense


def main() -> None:
    ensure_dirs()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_PATH:", DATASET_PATH)
    print("Exists:", DATASET_PATH.exists())

    df = load_dataset(DATASET_PATH)

    texts = df[TEXT_COL].astype(str)
    y = df["is_toxic"].values
    ids = df[ID_COL].values

    feature_union, dense_projector = build_feature_extractors()

    # ---------------- Holdout ----------------
    X_train_text, X_test_text, y_train, y_test, id_train, id_test = train_test_split(
        texts,
        y,
        ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test, fitted_union, X_train_dense, X_test_dense = extract_features(
        X_train_text,
        X_test_text,
        feature_union,
        dense_projector,
    )

    np.save(MODELS_DIR / "tier2_holdout_train_dense.npy", X_train_dense)
    np.save(MODELS_DIR / "tier2_holdout_test_dense.npy", X_test_dense)
    joblib.dump(fitted_union, MODELS_DIR / "tier2_feature_union.joblib")
    joblib.dump(dense_projector, MODELS_DIR / "tier2_svd.joblib")

    holdout_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    holdout_model.fit(X_train, y_train)

    holdout_prob = holdout_model.predict_proba(X_test)[:, 1]
    holdout_pred = (holdout_prob > 0.65).astype(int)

    joblib.dump(holdout_model, MODELS_DIR / "tier2_logreg_holdout.joblib")

    pd.DataFrame(
        {
            "id": id_test,
            "y_true": y_test,
            "y_pred": holdout_pred,
            "y_prob": holdout_prob,
        }
    ).to_csv(PREDICTIONS_DIR / "holdout_predictions.csv", index=False)

    # ---------------- Cross-validation ----------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(df), dtype=int)
    oof_prob = np.zeros(len(df), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), start=1):
        X_train_text_fold = texts.iloc[train_idx]
        X_val_text_fold = texts.iloc[val_idx]
        y_tr = y[train_idx]

        fold_feature_union, fold_dense_projector = build_feature_extractors()
        X_tr, X_val, _, _, _ = extract_features(
            X_train_text_fold,
            X_val_text_fold,
            fold_feature_union,
            fold_dense_projector,
        )

        fold_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        fold_model.fit(X_tr, y_tr)

        oof_prob[val_idx] = fold_model.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = (oof_prob[val_idx] > 0.65).astype(int)

        joblib.dump(fold_model, MODELS_DIR / f"tier2_logreg_fold_{fold}.joblib")

    pd.DataFrame(
        {
            "id": ids,
            "y_true": y,
            "y_pred": oof_pred,
            "y_prob": oof_prob,
        }
    ).to_csv(PREDICTIONS_DIR / "oof_predictions.csv", index=False)

    # ---------------- Final deployment model ----------------
    final_feature_union, final_dense_projector = build_feature_extractors()
    X_final_tfidf = final_feature_union.fit_transform(texts)
    X_final_dense = final_dense_projector.fit_transform(X_final_tfidf)
    X_final = hstack([X_final_tfidf, X_final_dense]).tocsr()

    final_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_final, y)
    joblib.dump(final_model, MODELS_DIR / "tier2_logreg_final.joblib")
    joblib.dump(final_feature_union, MODELS_DIR / "tier2_feature_union_final.joblib")
    joblib.dump(final_dense_projector, MODELS_DIR / "tier2_svd_final.joblib")
    np.save(MODELS_DIR / "tier2_dense_final.npy", X_final_dense)

    metadata = {
        "timestamp": utc_now(),
        "feature_pipeline": {
            "word_tfidf": {
                "ngram_range": [1, 2],
                "max_features": WORD_MAX_FEATURES,
                "stop_words": "english",
            },
            "char_tfidf": {
                "ngram_range": [3, 5],
                "max_features": CHAR_MAX_FEATURES,
                "analyzer": "char_wb",
            },
            "dense_projection": {
                "method": "TruncatedSVD",
                "components": SVD_COMPONENTS,
            },
        },
        "classifier": "LogisticRegression(class_weight='balanced')",
        "dataset_path": str(DATASET_PATH),
        "encoder_mode": "tfidf_plus_classic_dense_projection",
        "label_definition": "is_toxic = max(toxic, severe_toxic, obscene, threat, insult, identity_hate)",
    }
    with open(REPORTS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training artifacts saved.")

if __name__ == "__main__":
    main()