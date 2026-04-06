"""Baseline TF-IDF puro — sin augmentación, sin stacking, sin transformer.

Pipeline: TfidfVectorizer → LogisticRegression
Split: 80/20 estratificado (mismo que tus otros tests)
Genera reporte JSON compatible para la tabla comparativa.
"""

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent
CSV_PATH = _ROOT / "data" / "processed" / "train_clean_validated.csv"
REPORT_PATH = _ROOT / "data" / "processed" / "tfidf_baseline_report.json"


def main():
    t0 = time.perf_counter()
    print("=== BASELINE TF-IDF PURO ===\n")

    # --- Carga ---
    print(f"Cargando dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  → {len(df):,} filas")

    toxicity_cols = [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ]
    df["label"] = df[toxicity_cols].max(axis=1)

    X = df["comment_text"].astype(str)
    y = df["label"]

    # --- Split 80/20 estratificado ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # --- TF-IDF ---
    print("Ajustando TfidfVectorizer...")
    t_tfidf = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    tfidf_seconds = time.perf_counter() - t_tfidf
    print(f"  Features: {X_train_tfidf.shape[1]:,}")
    print(f"  TF-IDF tiempo: {tfidf_seconds:.2f}s\n")

    # --- Logistic Regression ---
    print("Entrenando LogisticRegression...")
    t_train = time.perf_counter()
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_tfidf, y_train)
    train_seconds = time.perf_counter() - t_train
    print(f"  Entrenamiento: {train_seconds:.2f}s\n")

    # --- Predicción ---
    print("Prediciendo...")
    t_pred = time.perf_counter()
    y_pred = clf.predict(X_test_tfidf)
    pred_seconds = time.perf_counter() - t_pred
    throughput = len(X_test) / pred_seconds
    print(f"  Predicción: {pred_seconds:.2f}s ({throughput:,.1f} filas/s)\n")

    # --- Métricas ---
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=" * 50)
    print("RESULTADOS — TF-IDF PURO")
    print("=" * 50)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")

    elapsed = time.perf_counter() - t0

    # --- Reporte JSON ---
    report = {
        "meta": {
            "script": "test_tfidf_baseline.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "tfidf_seconds": round(tfidf_seconds, 3),
            "training_seconds": round(train_seconds, 3),
            "prediction_seconds": round(pred_seconds, 3),
            "throughput_rows_per_sec": round(throughput, 1),
            "methodology": "tfidf_pure_baseline",
            "model": "LogisticRegression(solver='liblinear', class_weight='balanced')",
            "vectorizer": {
                "max_features": 50_000,
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "min_df": 3,
                "max_df": 0.95,
            },
        },
        "dataset": {
            "total_rows": len(df),
            "toxic_positives": int(y.sum()),
            "toxic_rate_pct": round(y.mean() * 100, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": X_train_tfidf.shape[1],
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "metrics": {
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReporte JSON en: {REPORT_PATH}")
    print(f"Tiempo total: {elapsed:.2f}s")


if __name__ == "__main__":
    main()