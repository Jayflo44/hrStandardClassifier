from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def evaluate_predictions(predictions_path: Path, report_name: str) -> dict:
    df = pd.read_csv(predictions_path)
    metrics = compute_metrics(df)

    report = {
        "timestamp": utc_now(),
        "predictions_file": str(predictions_path),
        "num_rows": int(len(df)),
        "metrics": metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / report_name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("PREDICTIONS_DIR:", PREDICTIONS_DIR)
    print("REPORTS_DIR:", REPORTS_DIR)

    holdout_path = PREDICTIONS_DIR / "holdout_predictions.csv"
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"


    print("Holdout exists:", holdout_path.exists())
    print("OOF exists:", oof_path.exists())
    
    if holdout_path.exists():
        holdout_report = evaluate_predictions(holdout_path, "holdout_evaluation.json")
        print("Holdout metrics:")
        print(json.dumps(holdout_report["metrics"], indent=2))

    if oof_path.exists():
        cv_report = evaluate_predictions(oof_path, "cv_evaluation.json")
        print("Cross-validation metrics:")
        print(json.dumps(cv_report["metrics"], indent=2))


if __name__ == "__main__":
    main()