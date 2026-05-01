from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "predictions" / "holdout_predictions.csv"
OUTPUT_CSV = PROJECT_ROOT / "artifacts" / "reports" / "threshold_sweep.csv"
OUTPUT_JSON = PROJECT_ROOT / "artifacts" / "reports" / "best_threshold.json"


def main() -> None:
    df = pd.read_csv(PREDICTIONS_PATH)

    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError("Prediction file must contain 'y_true' and 'y_prob' columns.")

    y_true = df["y_true"].astype(int).values
    y_prob = df["y_prob"].astype(float).values

    results = []

    thresholds = np.arange(0.50, 0.91, 0.01)

    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)

        row = {
            "threshold": round(float(threshold), 2),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "flagged_count": int(y_pred.sum()),
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    best_row = results_df.sort_values(
        by=["f1_score", "precision", "accuracy"],
        ascending=[False, False, False]
    ).iloc[0].to_dict()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    print("Top 10 thresholds by F1:")
    print(results_df.sort_values(by="f1_score", ascending=False).head(10).to_string(index=False))

    print("\nBest threshold:")
    print(json.dumps(best_row, indent=2))

    print(f"\nSaved full sweep to: {OUTPUT_CSV}")
    print(f"Saved best threshold to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()