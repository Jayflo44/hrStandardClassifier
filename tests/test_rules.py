"""Testing del Tier 1 Bouncer contra el dataset preprocesado.

Evaluación multi-label:
- Ground truth principal: any_toxic (1 si CUALQUIERA de las 6 etiquetas es 1).
- Desglose individual por cada clase (toxic, severe_toxic, obscene, etc.).
- Reporte JSON con métricas globales + por clase + ejemplos de errores.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from tier_one_bouncer import TierOneBouncer

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _ROOT / "hrStandardClassifier" / "data" / "processed" / "train_clean_validated.csv"
RULES_PATH = _ROOT / "hrStandardClassifier" /"hr_rules.yaml"
REPORT_PATH = _ROOT / "artifacts" / "reports" / "tier1_test_report.json"

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# ---------------------------------------------------------------------------
# Instancia única del bouncer
# ---------------------------------------------------------------------------

_bouncer = TierOneBouncer(RULES_PATH)


# ---------------------------------------------------------------------------
# Unit tests (pytest-compatible)
# ---------------------------------------------------------------------------


def test_tier_one_flags_lexicon_match() -> None:
    r = _bouncer.inspect("that is complete bullshit")
    assert r["status"] == "FLAGGED"
    assert r["reason"] == "Severe Profanity (NSFW)"


def test_tier_one_flags_hate_speech() -> None:
    r = _bouncer.inspect("you stupid nigger")
    assert r["status"] == "FLAGGED"
    assert r["reason"] == "Hate Speech and Slurs"


def test_tier_one_flags_obfuscated() -> None:
    r = _bouncer.inspect("f@ck you")
    assert r["status"] == "FLAGGED"
    assert r["reason"] == "Obfuscated Profanity"


def test_tier_one_flags_ssn() -> None:
    r = _bouncer.inspect("my ssn is 123-45-6789")
    assert r["status"] == "FLAGGED"
    assert r["reason"] == "PII Leak - US Social Security Number"


def test_tier_one_flags_credit_card() -> None:
    r = _bouncer.inspect("card 4111-1111-1111-1111")
    assert r["status"] == "FLAGGED"
    assert r["reason"] == "PII Leak - Credit Card (PCI-DSS)"


def test_tier_one_passes_neutral() -> None:
    r = _bouncer.inspect("the quarterly report looks fine")
    assert r["status"] == "PASS"


def test_tier_one_passes_professional() -> None:
    r = _bouncer.inspect("please review the attached document before friday")
    assert r["status"] == "PASS"


# ---------------------------------------------------------------------------
# Predicción batch — una sola pasada, guarda categoría para reusar
# ---------------------------------------------------------------------------


def _predict_batch(texts: pd.Series) -> tuple[list[int], list[str]]:
    """Predice en batch. Retorna (predicciones, categorías).

    Una sola pasada sobre los textos — las categorías se reusan
    para el desglose sin necesidad de una segunda pasada.
    """
    preds = []
    categories = []
    inspect = _bouncer.inspect  # pre-bind

    for t in texts:
        s = t if isinstance(t, str) else ""
        r = inspect(s)
        if r["status"] == "FLAGGED":
            preds.append(1)
            categories.append(r["reason"])
        else:
            preds.append(0)
            categories.append("")

    return preds, categories


# ---------------------------------------------------------------------------
# Métricas por clase
# ---------------------------------------------------------------------------


def _compute_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Calcula métricas binarias para una sola clase."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    n = len(y_true)
    return {
        "support_positive": int(y_true.sum()),
        "support_negative": int(n - y_true.sum()),
        "positive_rate_pct": round(100 * float(y_true.sum()) / n, 4),
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
            "accuracy": round(float((tp + tn) / n), 6),
        },
    }


# ---------------------------------------------------------------------------
# Evaluación completa
# ---------------------------------------------------------------------------


def run_evaluation(
    csv_path: Path = CSV_PATH,
    report_path: Path = REPORT_PATH,
) -> None:
    """Ejecuta evaluación multi-label y genera reporte JSON."""
    t0 = time.perf_counter()

    print(f"Cargando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    n_rows = len(df)
    print(f"  → {n_rows:,} filas")

    # --- Verificar columnas ---
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas faltantes en el dataset: {missing}")

    # --- Ground truth combinado ---
    df["any_toxic"] = df[LABEL_COLS].max(axis=1)

    # --- Predicciones (una sola pasada) ---
    print("Ejecutando predicciones Tier 1...")
    t_pred = time.perf_counter()
    preds, categories = _predict_batch(df["comment_text"])
    pred_time = time.perf_counter() - t_pred
    print(f"  → {pred_time:.2f}s ({n_rows / pred_time:,.0f} filas/s)")

    df["tier1_pred"] = preds
    df["tier1_category"] = categories
    y_pred = np.array(preds)

    # --- Desglose de categorías disparadas ---
    cat_series = df.loc[df["tier1_pred"] == 1, "tier1_category"]
    category_counts = cat_series.value_counts().to_dict()

    # =================================================================
    # EVALUACIÓN GLOBAL: any_toxic vs tier1_pred
    # =================================================================
    print("\n" + "=" * 60)
    print("EVALUACIÓN GLOBAL (any_toxic)")
    print("=" * 60)

    y_true_global = df["any_toxic"].values
    global_metrics = _compute_class_metrics(y_true_global, y_pred)
    cm_g = global_metrics["confusion_matrix"]

    print(f"                 Pred OK    Pred FLAGGED")
    print(f"  Real OK       {cm_g['true_negatives']:>8,}      {cm_g['false_positives']:>8,}")
    print(f"  Real TOXIC    {cm_g['false_negatives']:>8,}      {cm_g['true_positives']:>8,}")

    report_text = classification_report(
        y_true_global, y_pred,
        target_names=["OK (0)", "FLAGGED (1)"],
        zero_division=0,
    )
    print(report_text)

    # =================================================================
    # EVALUACIÓN POR CLASE
    # =================================================================
    print("=" * 60)
    print("DESGLOSE POR CLASE")
    print("=" * 60)

    per_class: dict[str, dict] = {}
    for col in LABEL_COLS:
        y_true_col = df[col].values
        metrics = _compute_class_metrics(y_true_col, y_pred)
        per_class[col] = metrics

        m = metrics["metrics"]
        cm_c = metrics["confusion_matrix"]
        print(
            f"  {col:20s} | "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1_score']:.4f} | "
            f"TP={cm_c['true_positives']:>5,}  FN={cm_c['false_negatives']:>5,}  "
            f"FP={cm_c['false_positives']:>5,}  | "
            f"support={metrics['support_positive']:,}"
        )

    # =================================================================
    # ANÁLISIS DE ERRORES
    # =================================================================

    # Falsos negativos globales: any_toxic=1 pero tier1 dijo PASS
    fn_mask = (y_true_global == 1) & (y_pred == 0)
    fp_mask = (y_true_global == 0) & (y_pred == 1)

    # ¿Qué etiquetas tenían los falsos negativos?
    fn_label_dist = {}
    if fn_mask.sum() > 0:
        fn_subset = df.loc[fn_mask, LABEL_COLS]
        for col in LABEL_COLS:
            fn_label_dist[col] = int(fn_subset[col].sum())

    fn_examples = (
        df.loc[fn_mask, "comment_text"]
        .head(15)
        .apply(lambda t: str(t)[:200])
        .tolist()
    )

    # ¿Qué categoría disparó los falsos positivos?
    fp_cat_dist = {}
    if fp_mask.sum() > 0:
        fp_cats = df.loc[fp_mask, "tier1_category"]
        fp_cat_dist = fp_cats.value_counts().to_dict()

    fp_examples = (
        df.loc[fp_mask, "comment_text"]
        .head(15)
        .apply(lambda t: str(t)[:200])
        .tolist()
    )

    print(f"\nFalsos Negativos (tóxicos no detectados): {int(fn_mask.sum()):,}")
    if fn_label_dist:
        print("  Distribución de etiquetas en FN:")
        for col, cnt in sorted(fn_label_dist.items(), key=lambda x: -x[1]):
            print(f"    {col:20s}: {cnt:,}")

    print(f"Falsos Positivos (limpios marcados):      {int(fp_mask.sum()):,}")
    if fp_cat_dist:
        print("  Categoría que los disparó:")
        for cat, cnt in sorted(fp_cat_dist.items(), key=lambda x: -x[1]):
            print(f"    {cat:40s}: {cnt:,}")

    if fn_examples:
        print("\nEjemplos de Falsos Negativos (primeros 5):")
        for i, ex in enumerate(fn_examples[:5], 1):
            print(f"  [{i}] {ex}")

    if fp_examples:
        print("\nEjemplos de Falsos Positivos (primeros 5):")
        for i, ex in enumerate(fp_examples[:5], 1):
            print(f"  [{i}] {ex}")

    # =================================================================
    # REPORTE JSON
    # =================================================================

    elapsed = time.perf_counter() - t0

    report_data = {
        "meta": {
            "script": "test_tier_one.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "prediction_seconds": round(pred_time, 3),
            "throughput_rows_per_sec": round(n_rows / pred_time, 1),
        },
        "files": {
            "dataset": str(csv_path),
            "rules": str(RULES_PATH),
            "report": str(report_path),
        },
        "dataset": {
            "total_rows": n_rows,
            "label_columns": LABEL_COLS,
            "any_toxic_positives": int(y_true_global.sum()),
            "any_toxic_negatives": int(n_rows - y_true_global.sum()),
            "any_toxic_rate_pct": round(
                100 * float(y_true_global.sum()) / n_rows, 4
            ),
            "per_label_positives": {
                col: int(df[col].sum()) for col in LABEL_COLS
            },
        },
        "global_evaluation": {
            "ground_truth": "any_toxic (OR of all 6 labels)",
            **global_metrics,
        },
        "per_class_evaluation": per_class,
        "category_breakdown": dict(
            sorted(category_counts.items(), key=lambda x: -x[1])
        ),
        "error_analysis": {
            "false_negatives": {
                "count": int(fn_mask.sum()),
                "label_distribution": fn_label_dist,
                "examples": fn_examples,
            },
            "false_positives": {
                "count": int(fp_mask.sum()),
                "triggered_categories": fp_cat_dist,
                "examples": fp_examples,
            },
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nReporte JSON en: {report_path}")
    print(f"Tiempo total: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()