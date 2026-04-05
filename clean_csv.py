"""Validación y limpieza post-preprocesamiento del dataset Jigsaw.

Genera un reporte técnico JSON con:
- Resultado de validación de integridad (IDs y etiquetas).
- Análisis de nulos por columna.
- Estadísticas del dataset limpio (distribución de labels, longitud de texto).
- Filas eliminadas y motivo.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent

RAW_CSV = (
    _ROOT
    / "data"
    / "raw"
    / "jigsaw-toxic-comment-classification-challenge"
    / "train"
    / "train.csv"
)
PREPROCESSED_CSV = _ROOT / "data" / "processed" / "train_clean.csv"
OUTPUT_CSV = _ROOT / "data" / "processed" / "train_clean_validated.csv"
REPORT_JSON = _ROOT / "data" / "processed" / "validation_report.json"

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

CHECK_COLS = ["id"] + LABEL_COLS


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_datasets(
    raw_path: Path, prep_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Cargando columnas de control del original: %s", raw_path)
    df_original = pd.read_csv(raw_path, usecols=CHECK_COLS)
    logger.info("  → %d filas", len(df_original))

    logger.info("Cargando dataset preprocesado: %s", prep_path)
    df_prep = pd.read_csv(prep_path)
    logger.info("  → %d filas, %d columnas", *df_prep.shape)

    return df_original, df_prep


def validate_integrity(
    df_original: pd.DataFrame, df_prep: pd.DataFrame
) -> dict:
    """Validación vectorizada. Retorna dict con resultados para el reporte."""
    logger.info("Validando integridad de IDs y etiquetas...")

    result: dict = {
        "passed": True,
        "original_rows": len(df_original),
        "preprocessed_rows": len(df_prep),
        "columns_checked": CHECK_COLS,
        "missing_columns": [],
        "row_count_match": len(df_original) == len(df_prep),
        "column_mismatches": {},
    }

    missing = [c for c in CHECK_COLS if c not in df_prep.columns]
    if missing:
        result["missing_columns"] = missing
        result["passed"] = False
        logger.error("Columnas faltantes: %s", missing)
        return result

    if not result["row_count_match"]:
        result["passed"] = False
        logger.error(
            "Filas: original=%d, preprocesado=%d",
            len(df_original), len(df_prep),
        )
        return result

    for col in CHECK_COLS:
        orig_vals = df_original[col].values
        prep_vals = df_prep[col].values

        if orig_vals.dtype.kind in ("U", "S", "O"):
            mismatches = int(np.sum(orig_vals != prep_vals))
        else:
            if np.array_equal(orig_vals, prep_vals):
                continue
            mismatches = int(np.sum(orig_vals != prep_vals))

        if mismatches > 0:
            result["column_mismatches"][col] = mismatches
            result["passed"] = False
            logger.error("  ❌ '%s': %d diferencias", col, mismatches)

    if result["passed"]:
        logger.info("  ✅ IDs y etiquetas coinciden perfectamente.")

    return result


def analyze_nulls(df: pd.DataFrame) -> dict:
    """Analiza nulos y retorna dict para el reporte."""
    null_counts = df.isnull().sum()
    total = len(df)

    per_column = {}
    for col in df.columns:
        count = int(null_counts[col])
        if count > 0:
            per_column[col] = {
                "count": count,
                "percentage": round(100 * count / total, 4),
            }

    rows_with_any_null = int(df.isnull().any(axis=1).sum())

    report = {
        "total_nulls": int(null_counts.sum()),
        "rows_with_any_null": rows_with_any_null,
        "per_column": per_column,
    }

    if per_column:
        logger.warning("Nulos encontrados:")
        for col, info in per_column.items():
            logger.warning(
                "  %-20s %d (%.2f%%)", col, info["count"], info["percentage"]
            )
    else:
        logger.info("Sin valores nulos en el dataset.")

    return report


def clean_nulls(
    df: pd.DataFrame, text_column: str = "comment_text"
) -> tuple[pd.DataFrame, dict]:
    """Elimina filas con nulos en text_column. Retorna df limpio + info."""
    before = len(df)
    null_mask = df[text_column].isna()
    n_dropped = int(null_mask.sum())

    dropped_ids = []
    if n_dropped > 0:
        dropped_ids = df.loc[null_mask, "id"].tolist() if "id" in df.columns else []
        df = df.loc[~null_mask]
        logger.info("Eliminadas %d filas con nulos en '%s'", n_dropped, text_column)

    clean_info = {
        "rows_before": before,
        "rows_dropped": n_dropped,
        "rows_after": len(df),
        "drop_reason": f"null in '{text_column}'",
        "dropped_ids_sample": dropped_ids[:50],  # máximo 50 IDs de muestra
    }

    return df, clean_info


def compute_dataset_stats(
    df: pd.DataFrame, text_column: str = "comment_text"
) -> dict:
    """Estadísticas del dataset limpio para el reporte."""
    lengths = df[text_column].str.len()

    label_stats = {}
    for col in LABEL_COLS:
        if col in df.columns:
            positives = int(df[col].sum())
            label_stats[col] = {
                "positive": positives,
                "negative": len(df) - positives,
                "positive_rate": round(100 * positives / len(df), 4),
            }

    # Textos vacíos post-limpieza
    empty_texts = int((df[text_column].str.strip() == "").sum())

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "empty_texts": empty_texts,
        "text_length": {
            "mean": round(float(lengths.mean()), 2),
            "median": round(float(lengths.median()), 2),
            "std": round(float(lengths.std()), 2),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "p25": round(float(lengths.quantile(0.25)), 2),
            "p75": round(float(lengths.quantile(0.75)), 2),
            "p95": round(float(lengths.quantile(0.95)), 2),
        },
        "label_distribution": label_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()

    report: dict = {
        "meta": {
            "script": "validate_clean.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "files": {
            "raw_csv": str(RAW_CSV),
            "preprocessed_csv": str(PREPROCESSED_CSV),
            "output_csv": str(OUTPUT_CSV),
            "report_json": str(REPORT_JSON),
        },
    }

    # 1. Cargar
    df_original, df_prep = load_datasets(RAW_CSV, PREPROCESSED_CSV)

    # 2. Validar integridad
    integrity = validate_integrity(df_original, df_prep)
    report["integrity_validation"] = integrity

    if not integrity["passed"]:
        report["status"] = "ABORTED"
        report["meta"]["elapsed_seconds"] = round(
            time.perf_counter() - t0, 3
        )
        _save_report(report)
        logger.error("Abortando: integridad comprometida.")
        sys.exit(1)

    del df_original

    # 3. Analizar nulos
    report["null_analysis"] = analyze_nulls(df_prep)

    # 4. Limpiar nulos
    df_clean, clean_info = clean_nulls(df_prep)
    report["cleaning"] = clean_info

    del df_prep

    # 5. Verificación final
    remaining = int(df_clean.isnull().sum().sum())
    report["post_clean_remaining_nulls"] = remaining

    if remaining:
        logger.warning("Aún quedan %d nulos.", remaining)
    else:
        logger.info("✅ Dataset limpio, sin nulos.")

    # 6. Estadísticas del dataset limpio
    report["dataset_stats"] = compute_dataset_stats(df_clean)

    # 7. Guardar CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_CSV, index=False)
    logger.info("Dataset guardado en: %s", OUTPUT_CSV)

    # 8. Guardar reporte
    elapsed = time.perf_counter() - t0
    report["meta"]["elapsed_seconds"] = round(elapsed, 3)
    report["status"] = "OK"

    _save_report(report)
    print(f"\nCSV validado en:   {OUTPUT_CSV}")
    print(f"Reporte JSON en:   {REPORT_JSON}")
    print(f"Tiempo total: {elapsed:.2f}s")


def _save_report(report: dict) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Reporte técnico guardado en %s", REPORT_JSON)


if __name__ == "__main__":
    main()