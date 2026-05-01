from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "hrClassifierSrc" / "data" / "processed" / "train_clean_validated.csv"
TESTSET_PATH = PROJECT_ROOT / "hrClassifierSrc" / "data" / "processed" / "test_clean_validated.csv"
DECISION_THRESHOLD = 0.95
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TRAIN_BERT_EMBEDDINGS_PATH = MODELS_DIR / "tier2_train_bert_embeddings.npy"
TEST_BERT_EMBEDDINGS_PATH = MODELS_DIR / "tier2_test_bert_embeddings.npy"

BERT_MODEL_NAME = "unitary/toxic-bert"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 32

HARD_EXAMPLES_DIR = ARTIFACTS_DIR / "hard_examples"
HARD_EXAMPLE_PROB_MIN = 0.40
HARD_EXAMPLE_PROB_MAX = DECISION_THRESHOLD
TEXT_COL = "comment_text"
ID_COL = "id"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

RANDOM_STATE = 42
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
    HARD_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

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
        min_df=2  
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 7),
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

# def load_bert_encoder():
#     tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
#     model = AutoModel.from_pretrained(BERT_MODEL_NAME)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     return tokenizer, model, device

# def extract_bert_embeddings(
#     texts: pd.Series,
#     tokenizer,
#     model,
#     device,
# ) -> np.ndarray:
#     text_list = texts.fillna("").astype(str).tolist()
#     all_embeddings = []

#     for start in range(0, len(text_list), BERT_BATCH_SIZE):
#         batch_texts = text_list[start:start + BERT_BATCH_SIZE]

#         inputs = tokenizer(
#             batch_texts,
#             padding=True,
#             truncation=True,
#             max_length=BERT_MAX_LENGTH,
#             return_tensors="pt",
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)

#         cls_embeddings = outputs.last_hidden_state[:, 0, :]
#         all_embeddings.append(cls_embeddings.cpu().numpy())

#     return np.vstack(all_embeddings)


# def save_hard_example_embeddings(
#     predictions_df: pd.DataFrame,
#     source_name: str,
# ) -> dict:
#     hard_df = predictions_df[
#         (predictions_df["y_true"] != predictions_df["y_pred"])
#         & (predictions_df["y_prob"] >= HARD_EXAMPLE_PROB_MIN)
#         & (predictions_df["y_prob"] <= HARD_EXAMPLE_PROB_MAX)
#     ].copy()

#     report = {
#         "source": source_name,
#         "rows_scored": int(len(predictions_df)),
#         "hard_examples_found": int(len(hard_df)),
#         "probability_window": {
#             "min": HARD_EXAMPLE_PROB_MIN,
#             "max": HARD_EXAMPLE_PROB_MAX,
#         },
#     }

#     if hard_df.empty:
#         return report

#     tokenizer, bert_model, device = load_bert_encoder()

#     embeddings = extract_bert_embeddings(
#         hard_df[TEXT_COL],
#         tokenizer,
#         bert_model,
#         device,
#     )

#     metadata_path = HARD_EXAMPLES_DIR / f"{source_name}_hard_examples.csv"
#     embeddings_path = HARD_EXAMPLES_DIR / f"{source_name}_hard_bert_embeddings.npy"

#     hard_df.to_csv(metadata_path, index=False)
#     np.save(embeddings_path, embeddings)

#     report["metadata_path"] = str(metadata_path)
#     report["embeddings_path"] = str(embeddings_path)
#     report["embedding_shape"] = list(embeddings.shape)

#     return report

def main() -> None:
    ensure_dirs()
    print("TESTSET_PATH:", TESTSET_PATH)
    print("Test exists:", TESTSET_PATH.exists())
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_PATH:", DATASET_PATH)
    print("Exists:", DATASET_PATH.exists())

    train_df = load_dataset(DATASET_PATH)
    test_df = load_dataset(TESTSET_PATH)

    train_texts = train_df[TEXT_COL].astype(str)
    y_train_full = train_df["is_toxic"].values
    train_ids = train_df[ID_COL].values

    test_texts = test_df[TEXT_COL].astype(str)
    y_test_external = test_df["is_toxic"].values
    test_ids = test_df[ID_COL].values

    feature_union, dense_projector = build_feature_extractors()

    # ---------------- External Test Evaluation ----------------
    X_train_full, X_test_external, fitted_union, X_train_dense, X_test_dense = extract_features(
        train_texts,
        test_texts,
        feature_union,
        dense_projector,
    )
    X_train_bert = np.load(TRAIN_BERT_EMBEDDINGS_PATH)
    X_test_bert = np.load(TEST_BERT_EMBEDDINGS_PATH)

    if X_train_bert.shape[0] != X_train_full.shape[0]:
        raise ValueError("Train BERT embeddings row count does not match train features.")

    if X_test_bert.shape[0] != X_test_external.shape[0]:
        raise ValueError("Test BERT embeddings row count does not match test features.")

    X_train_full = hstack([X_train_full, csr_matrix(X_train_bert)]).tocsr()
    X_test_external = hstack([X_test_external, csr_matrix(X_test_bert)]).tocsr()

    joblib.dump(fitted_union, MODELS_DIR / "tier2_feature_union.joblib")
    joblib.dump(dense_projector, MODELS_DIR / "tier2_svd.joblib")

    holdout_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    holdout_model.fit(X_train_full, y_train_full)

    holdout_prob = holdout_model.predict_proba(X_test_external)[:, 1]
    holdout_pred = (holdout_prob > DECISION_THRESHOLD).astype(int)

    joblib.dump(holdout_model, MODELS_DIR / "tier2_logreg_holdout.joblib")

    test_predictions_df = pd.DataFrame(
    {
        "id": test_ids,
        "comment_text": test_texts.values,
        "y_true": y_test_external,
        "y_pred": holdout_pred,
        "y_prob": holdout_prob,
    }
    )

    test_predictions_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    # test_hard_report = save_hard_example_embeddings(
    #     test_predictions_df,
    #     source_name="test",
    # )

    # ---------------- Cross-validation ----------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(train_df), dtype=int)
    oof_prob = np.zeros(len(train_df), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, y_train_full), start=1):
        X_train_text_fold = train_texts.iloc[train_idx]
        X_val_text_fold = train_texts.iloc[val_idx]
        y_tr = y_train_full[train_idx]
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
        oof_pred[val_idx] = (oof_prob[val_idx] > DECISION_THRESHOLD).astype(int)

        joblib.dump(fold_model, MODELS_DIR / f"tier2_logreg_fold_{fold}.joblib")

    oof_predictions_df = pd.DataFrame(
    {
        "id": train_ids,
        "comment_text": train_texts.values,
        "y_true": y_train_full,
        "y_pred": oof_pred,
        "y_prob": oof_prob,
    }
    )
    oof_predictions_df.to_csv(PREDICTIONS_DIR / "oof_predictions.csv", index=False)
    # oof_hard_report = save_hard_example_embeddings(
    #     oof_predictions_df,
    #     source_name="oof",
    # )
    # ---------------- Final deployment model ----------------
    final_feature_union, final_dense_projector = build_feature_extractors()
    X_final_tfidf = final_feature_union.fit_transform(train_texts)
    X_final_dense = final_dense_projector.fit_transform(X_final_tfidf)
    X_final = hstack([
        X_final_tfidf,
        csr_matrix(X_final_dense),
        csr_matrix(X_train_bert),
    ]).tocsr()
    final_model = LogisticRegression(
        class_weight={0: 1, 1: 1.3},
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_final, y_train_full)
    joblib.dump(final_model, MODELS_DIR / "tier2_logreg_final.joblib")
    joblib.dump(final_feature_union, MODELS_DIR / "tier2_feature_union_final.joblib")
    joblib.dump(final_dense_projector, MODELS_DIR / "tier2_svd_final.joblib")
    #np.save(MODELS_DIR / "tier2_dense_final.npy", X_final_dense)
        # ---------------- Final model predictions on external test set ----------------
    X_final_test_tfidf = final_feature_union.transform(test_texts)
    X_final_test_dense = final_dense_projector.transform(X_final_test_tfidf)
    X_final_test = hstack([
        X_final_test_tfidf,
        csr_matrix(X_final_test_dense),
        csr_matrix(X_test_bert),
    ]).tocsr()
    final_test_prob = final_model.predict_proba(X_final_test)[:, 1]
    final_test_pred = (final_test_prob > DECISION_THRESHOLD).astype(int)

    pd.DataFrame(
        {
            "id": test_ids,
            "y_true": y_test_external,
            "y_pred": final_test_pred,
            "y_prob": final_test_prob,
        }
    ).to_csv(PREDICTIONS_DIR / "final_model_test_predictions.csv", index=False)
    metadata = {
        "timestamp": utc_now(),
        "feature_pipeline": {
            "word_tfidf": {
                "ngram_range": [1, 3],
                "max_features": WORD_MAX_FEATURES,
                "stop_words": "english",
            },
            "char_tfidf": {
                "ngram_range": [3, 7],
                "max_features": CHAR_MAX_FEATURES,
                "analyzer": "char_wb",
            },
            "dense_projection": {
                "method": "TruncatedSVD",
                "components": SVD_COMPONENTS,
            },
            "hard_example_mining": {
            "bert_model_name": BERT_MODEL_NAME,
            "embedding_type": "cls_token",
            "max_length": BERT_MAX_LENGTH,
            "batch_size": BERT_BATCH_SIZE,
            "probability_window": {
                "min": HARD_EXAMPLE_PROB_MIN,
                "max": HARD_EXAMPLE_PROB_MAX,
            },
            # "test_hard_examples": test_hard_report,
            # "oof_hard_examples": oof_hard_report,
        },
        },
        "classifier": "LogisticRegression(class_weight='balanced')",
        "dataset_path": str(DATASET_PATH),
        "encoder_mode": "tfidf_plus_svd_plus_saved_bert_embeddings",        "label_definition": "is_toxic = max(toxic, severe_toxic, obscene, threat, insult, identity_hate)",
    }
    with open(REPORTS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training artifacts saved.")

if __name__ == "__main__":
    main()