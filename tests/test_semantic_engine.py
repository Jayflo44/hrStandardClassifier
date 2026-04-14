"""Tier 2 — Evaluación sobre los errores del Tier 1.

Flujo:
  1. Carga el dataset validado.
  2. Corre las predicciones del Tier 1 para identificar FP y FN.
  3. Pasa esos errores al Tier 2 (toxic-bert).
  4. Genera reporte JSON con métricas de recuperación.

Genera: data/processed/tier2_test_report.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from TierOneBouncer import TierOneBouncer
from TierTwoSemanticEngine import Tier2SemanticEngine

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
CSV_PATH = _ROOT / "data" / "processed" / "train_clean_validated.csv"
RULES_PATH = _ROOT / "hr_rules.yaml"
REPORT_PATH = _ROOT / "data" / "processed" / "tier2_test_report.json"
ERRORS_CSV_PATH = _ROOT / "data" / "processed" / "tier1_errors.csv"

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


# ---------------------------------------------------------------------------
# Tier 1 prediction (reusa la misma lógica de test_rules.py)
# ---------------------------------------------------------------------------


def _run_tier1_predictions(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Corre Tier 1 sobre el dataset y retorna predicciones + categorías."""
    bouncer = TierOneBouncer(RULES_PATH)
    inspect = bouncer.inspect

    preds = []
    categories = []
    for t in df["comment_text"]:
        s = t if isinstance(t, str) else ""
        r = inspect(s)
        if r["status"] == "FLAGGED":
            preds.append(1)
            categories.append(r["reason"])
        else:
            preds.append(0)
            categories.append("")

    return np.array(preds), categories


# ---------------------------------------------------------------------------
# Métricas helper
# ---------------------------------------------------------------------------


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    n = len(y_true)
    return {
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "metrics": {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1_score": round(float(f1), 6),
            "accuracy": round(float((tp + tn) / n), 6) if n > 0 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------


def run_tier2_evaluation(
    csv_path: Path = CSV_PATH,
    report_path: Path = REPORT_PATH,
    errors_csv_path: Path = ERRORS_CSV_PATH,
    batch_size: int = 32,
) -> None:
    """Identifica errores del Tier 1 y evalúa el Tier 2 sobre ellos."""
    t0 = time.perf_counter()

    # =================================================================
    # PASO 1: Cargar dataset y correr Tier 1
    # =================================================================

    print(f"Cargando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"  → {n_total:,} filas")

    df["any_toxic"] = df[LABEL_COLS].max(axis=1)

    print("Ejecutando predicciones Tier 1...")
    t1_start = time.perf_counter()
    tier1_preds, tier1_cats = _run_tier1_predictions(df)
    t1_time = time.perf_counter() - t1_start
    print(f"  → {t1_time:.2f}s")

    y_true = df["any_toxic"].values

    # =================================================================
    # PASO 2: Identificar FP y FN del Tier 1
    # =================================================================

    fn_mask = (y_true == 1) & (tier1_preds == 0)  # tóxicos no detectados
    fp_mask = (y_true == 0) & (tier1_preds == 1)  # limpios marcados
    error_mask = fn_mask | fp_mask

    n_fn = int(fn_mask.sum())
    n_fp = int(fp_mask.sum())
    n_errors = int(error_mask.sum())

    print(f"\nErrores del Tier 1:")
    print(f"  Falsos Negativos: {n_fn:,}")
    print(f"  Falsos Positivos: {n_fp:,}")
    print(f"  Total a evaluar:  {n_errors:,}")

    # Extraer subset de errores
    df_errors = df.loc[error_mask].copy()
    df_errors["tier1_error_type"] = np.where(
        fn_mask[error_mask], "false_negative", "false_positive"
    )
    df_errors["tier1_category"] = [
        tier1_cats[i] for i in range(n_total) if error_mask[i]
    ]

    # Exportar CSV de errores
    errors_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_errors.to_csv(errors_csv_path, index=False)
    print(f"  → Errores exportados a: {errors_csv_path}")

    # =================================================================
    # PASO 3: Correr Tier 2 sobre los errores
    # =================================================================

    engine = Tier2SemanticEngine()

    print(f"\nEjecutando Tier 2 sobre {n_errors:,} textos...")
    t2_start = time.perf_counter()

    probs = []
    decisions = []
    texts = df_errors["comment_text"].tolist()

    for i in range(0, n_errors, batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            t = text if isinstance(text, str) else ""
            r = engine.analyze_and_route(t)
            probs.append(r["toxic_prob"])
            decisions.append(r["decision"])

        done = min(i + batch_size, n_errors)
        if done % (batch_size * 20) == 0 or done == n_errors:
            elapsed_so_far = time.perf_counter() - t2_start
            rate = done / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"  → {done:,}/{n_errors:,} ({rate:.0f} filas/s)")

    t2_time = time.perf_counter() - t2_start

    df_errors["tier2_toxic_prob"] = probs
    df_errors["tier2_decision"] = decisions
    df_errors["tier2_pred"] = (
        df_errors["tier2_toxic_prob"] >= engine.threshold_toxic
    ).astype(int)

    y_true_errors = df_errors["any_toxic"].values
    y_pred_t2 = df_errors["tier2_pred"].values

    # =================================================================
    # PASO 4: Métricas globales sobre errores
    # =================================================================

    global_metrics = _compute_metrics(y_true_errors, y_pred_t2)
    cm_g = global_metrics["confusion_matrix"]

    print(f"\n{'=' * 60}")
    print("TIER 2 — EVALUACIÓN SOBRE ERRORES DEL TIER 1")
    print(f"{'=' * 60}")
    print(f"                 Pred OK    Pred FLAGGED")
    print(f"  Real OK       {cm_g['true_negatives']:>8,}      {cm_g['false_positives']:>8,}")
    print(f"  Real TOXIC    {cm_g['false_negatives']:>8,}      {cm_g['true_positives']:>8,}")
    m = global_metrics["metrics"]
    print(f"\n  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1_score']:.4f}")
    print(f"  Accuracy:  {m['accuracy']:.4f}")

    # =================================================================
    # PASO 5: Desglose — ¿Qué hizo el Tier 2 con cada tipo de error?
    # =================================================================

    fn_orig_mask = df_errors["tier1_error_type"] == "false_negative"
    fp_orig_mask = df_errors["tier1_error_type"] == "false_positive"

    # FN del Tier 1: tóxicos no detectados → ¿cuántos recuperó el Tier 2?
    fn_recovered = int((df_errors.loc[fn_orig_mask, "tier2_pred"] == 1).sum())
    fn_still_missed = n_fn - fn_recovered

    # FP del Tier 1: limpios marcados → ¿cuántos corrigió el Tier 2?
    fp_corrected = int((df_errors.loc[fp_orig_mask, "tier2_pred"] == 0).sum())
    fp_still_wrong = n_fp - fp_corrected

    print(f"\n{'=' * 60}")
    print("RESOLUCIÓN DE ERRORES DEL TIER 1")
    print(f"{'=' * 60}")
    print(f"\n  Falsos Negativos del Tier 1 ({n_fn:,} tóxicos no detectados):")
    print(f"    → Tier 2 recuperó (FLAGGED):  {fn_recovered:,} ({100*fn_recovered/max(n_fn,1):.1f}%)")
    print(f"    → Tier 2 también falló:       {fn_still_missed:,} ({100*fn_still_missed/max(n_fn,1):.1f}%)")

    print(f"\n  Falsos Positivos del Tier 1 ({n_fp:,} limpios marcados):")
    print(f"    → Tier 2 corrigió (OK):       {fp_corrected:,} ({100*fp_corrected/max(n_fp,1):.1f}%)")
    print(f"    → Tier 2 también erró:        {fp_still_wrong:,} ({100*fp_still_wrong/max(n_fp,1):.1f}%)")

    # Métricas separadas por tipo de error
    fn_metrics = {}
    if n_fn > 0:
        fn_df = df_errors.loc[fn_orig_mask]
        fn_metrics = _compute_metrics(
            fn_df["any_toxic"].values, fn_df["tier2_pred"].values
        )

    fp_metrics = {}
    if n_fp > 0:
        fp_df = df_errors.loc[fp_orig_mask]
        fp_metrics = _compute_metrics(
            fp_df["any_toxic"].values, fp_df["tier2_pred"].values
        )

    # =================================================================
    # PASO 6: Distribución de probabilidades + decisiones
    # =================================================================

    prob_stats = {
        "overall": {
            "mean": round(float(df_errors["tier2_toxic_prob"].mean()), 4),
            "median": round(float(df_errors["tier2_toxic_prob"].median()), 4),
            "std": round(float(df_errors["tier2_toxic_prob"].std()), 4),
            "min": round(float(df_errors["tier2_toxic_prob"].min()), 4),
            "max": round(float(df_errors["tier2_toxic_prob"].max()), 4),
        },
    }
    if n_fn > 0:
        fn_probs = df_errors.loc[fn_orig_mask, "tier2_toxic_prob"]
        prob_stats["false_negatives_from_tier1"] = {
            "mean": round(float(fn_probs.mean()), 4),
            "median": round(float(fn_probs.median()), 4),
            "std": round(float(fn_probs.std()), 4),
        }
    if n_fp > 0:
        fp_probs = df_errors.loc[fp_orig_mask, "tier2_toxic_prob"]
        prob_stats["false_positives_from_tier1"] = {
            "mean": round(float(fp_probs.mean()), 4),
            "median": round(float(fp_probs.median()), 4),
            "std": round(float(fp_probs.std()), 4),
        }

    decision_dist = df_errors["tier2_decision"].value_counts().to_dict()

    # Ejemplos residuales
    still_missed_mask = fn_orig_mask & (df_errors["tier2_pred"] == 0)
    still_missed_examples = (
        df_errors.loc[still_missed_mask, "comment_text"]
        .head(10)
        .apply(lambda t: str(t)[:200])
        .tolist()
    )

    both_fp_mask = fp_orig_mask & (df_errors["tier2_pred"] == 1)
    both_fp_examples = (
        df_errors.loc[both_fp_mask, "comment_text"]
        .head(10)
        .apply(lambda t: str(t)[:200])
        .tolist()
    )

    if still_missed_examples:
        print(f"\nEjemplos que ni Tier 1 ni Tier 2 detectaron:")
        for i, ex in enumerate(still_missed_examples[:5], 1):
            print(f"  [{i}] {ex}")

    # =================================================================
    # REPORTE JSON
    # =================================================================

    elapsed = time.perf_counter() - t0

    report_data = {
        "meta": {
            "script": "test_tier2.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "tier1_prediction_seconds": round(t1_time, 3),
            "tier2_prediction_seconds": round(t2_time, 3),
            "tier2_throughput_rows_per_sec": round(n_errors / t2_time, 1)
            if t2_time > 0
            else 0,
            "model": engine.model_name,
            "threshold_safe": engine.threshold_safe,
            "threshold_toxic": engine.threshold_toxic,
        },
        "files": {
            "dataset": str(csv_path),
            "rules": str(RULES_PATH),
            "errors_csv": str(errors_csv_path),
            "report": str(report_path),
        },
        "tier1_summary": {
            "total_rows_evaluated": n_total,
            "false_negatives": n_fn,
            "false_positives": n_fp,
            "total_errors": n_errors,
        },
        "tier2_global_evaluation": global_metrics,
        "tier1_error_resolution": {
            "false_negatives": {
                "total": n_fn,
                "recovered_by_tier2": fn_recovered,
                "still_missed": fn_still_missed,
                "recovery_rate_pct": round(
                    100 * fn_recovered / max(n_fn, 1), 2
                ),
                "metrics": fn_metrics,
            },
            "false_positives": {
                "total": n_fp,
                "corrected_by_tier2": fp_corrected,
                "still_wrong": fp_still_wrong,
                "correction_rate_pct": round(
                    100 * fp_corrected / max(n_fp, 1), 2
                ),
                "metrics": fp_metrics,
            },
        },
        "tier2_decision_distribution": decision_dist,
        "probability_distribution": prob_stats,
        "residual_errors": {
            "still_missed_count": int(still_missed_mask.sum()),
            "still_missed_examples": still_missed_examples,
            "both_tiers_fp_count": int(both_fp_mask.sum()),
            "both_tiers_fp_examples": both_fp_examples,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nReporte JSON en: {report_path}")
    print(f"Errores CSV en:  {errors_csv_path}")
    print(f"Tiempo total: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tier2_evaluation()