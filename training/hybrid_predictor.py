from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from transformers import pipeline as hf_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "hrClassifierSrc"
    / "data"
    / "processed"
    / "test_clean_audited.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

LR_MODEL_PATH = MODELS_DIR / "tier2_logreg_final.joblib"
FEATURE_UNION_PATH = MODELS_DIR / "tier2_feature_union_final.joblib"
SVD_PATH = MODELS_DIR / "tier2_svd_final.joblib"

TEXT_COL = "comment_text"
ID_COL = "id"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Hybrid thresholds
LR_SAFE_THRESHOLD = 0.30
LR_TOXIC_THRESHOLD = 0.90
BERT_TOXIC_THRESHOLD = 0.80

BERT_MODEL_PATH = MODELS_DIR / "toxic_bert_local"

OUTPUT_PREDICTIONS = PREDICTIONS_DIR / "hybrid_test_predictions.csv"
OUTPUT_REPORT = REPORTS_DIR / "hybrid_prediction_report.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    required_cols = [ID_COL, TEXT_COL] + LABEL_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df["is_toxic"] = df[LABEL_COLS].max(axis=1).astype(int)

    return df


def load_lr_artifacts() -> tuple[object, object, object]:
    if not LR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing LR model: {LR_MODEL_PATH}")
    if not FEATURE_UNION_PATH.exists():
        raise FileNotFoundError(f"Missing feature union: {FEATURE_UNION_PATH}")
    if not SVD_PATH.exists():
        raise FileNotFoundError(f"Missing SVD model: {SVD_PATH}")

    lr_model = joblib.load(LR_MODEL_PATH)
    feature_union = joblib.load(FEATURE_UNION_PATH)
    dense_projector = joblib.load(SVD_PATH)

    return lr_model, feature_union, dense_projector


def load_bert_pipeline():
    if not BERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing local BERT model: {BERT_MODEL_PATH}")

    return hf_pipeline(
        "text-classification",
        model=str(BERT_MODEL_PATH),
        tokenizer=str(BERT_MODEL_PATH),
        top_k=None,
    )

def get_bert_toxic_prob(classifier, text: str) -> float:
    text = str(text or "")
    raw_results = classifier(text[:512])

    if isinstance(raw_results[0], list):
        results = raw_results[0]
    else:
        results = raw_results

    for item in results:
        if item["label"].lower() == "toxic":
            return float(item["score"])

    return 0.0


def build_lr_features(df: pd.DataFrame, feature_union, dense_projector):
    texts = df[TEXT_COL].astype(str)
    X_tfidf = feature_union.transform(texts)
    X_dense = dense_projector.transform(X_tfidf)
    X_final = hstack([X_tfidf, X_dense]).tocsr()
    return X_final


def run_hybrid_prediction(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    lr_model, feature_union, dense_projector = load_lr_artifacts()
    bert_classifier = load_bert_pipeline()

    X_final = build_lr_features(df, feature_union, dense_projector)
    lr_probs = lr_model.predict_proba(X_final)[:, 1]

    final_preds: list[int] = []
    final_probs: list[float] = []
    decision_source: list[str] = []
    bert_probs: list[float | None] = []

    lr_safe_count = 0
    lr_toxic_count = 0
    bert_fallback_count = 0

    for text, lr_prob in zip(df[TEXT_COL].tolist(), lr_probs):
        if lr_prob < LR_SAFE_THRESHOLD:
            final_preds.append(0)
            final_probs.append(float(lr_prob))
            decision_source.append("lr_safe")
            bert_probs.append(None)
            lr_safe_count += 1

        elif lr_prob > LR_TOXIC_THRESHOLD:
            final_preds.append(1)
            final_probs.append(float(lr_prob))
            decision_source.append("lr_toxic")
            bert_probs.append(None)
            lr_toxic_count += 1

        else:
            bert_prob = get_bert_toxic_prob(bert_classifier, text)
            bert_pred = int(bert_prob > BERT_TOXIC_THRESHOLD)

            final_preds.append(bert_pred)
            final_probs.append(float(bert_prob))
            decision_source.append("bert_fallback")
            bert_probs.append(float(bert_prob))
            bert_fallback_count += 1

    output_df = pd.DataFrame(
        {
            "id": df[ID_COL].values,
            "comment_text": df[TEXT_COL].values,
            "y_true": df["is_toxic"].values,
            "lr_prob": lr_probs,
            "bert_prob": bert_probs,
            "y_pred": final_preds,
            "y_prob": final_probs,
            "decision_source": decision_source,
        }
    )

    report = {
        "timestamp": utc_now(),
        "dataset_path": str(DATASET_PATH),
        "rows_scored": int(len(df)),
        "lr_safe_threshold": LR_SAFE_THRESHOLD,
        "lr_toxic_threshold": LR_TOXIC_THRESHOLD,
        "bert_toxic_threshold": BERT_TOXIC_THRESHOLD,
        "bert_model_path": str(BERT_MODEL_PATH),
        "decision_breakdown": {
            "lr_safe": lr_safe_count,
            "lr_toxic": lr_toxic_count,
            "bert_fallback": bert_fallback_count,
        },
        "decision_breakdown_percent": {
            "lr_safe": round(100 * lr_safe_count / len(df), 2),
            "lr_toxic": round(100 * lr_toxic_count / len(df), 2),
            "bert_fallback": round(100 * bert_fallback_count / len(df), 2),
        },
        "artifacts_used": {
            "lr_model": str(LR_MODEL_PATH),
            "feature_union": str(FEATURE_UNION_PATH),
            "svd": str(SVD_PATH),
        },
    }

    return output_df, report


def save_outputs(pred_df: pd.DataFrame, report: dict) -> None:
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Hybrid predictions saved to: {OUTPUT_PREDICTIONS}")
    print(f"Hybrid report saved to: {OUTPUT_REPORT}")
    print(f"Rows scored: {len(pred_df)}")


def main() -> None:
    ensure_dirs()

    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("Running hybrid LR + BERT predictions...")
    pred_df, report = run_hybrid_prediction(df)

    print("Saving outputs...")
    save_outputs(pred_df, report)

    print("\nDecision summary:")
    print(json.dumps(report["decision_breakdown"], indent=2))
    print(json.dumps(report["decision_breakdown_percent"], indent=2))


if __name__ == "__main__":
    main()