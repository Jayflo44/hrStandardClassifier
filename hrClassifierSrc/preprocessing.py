"""Carga y preprocesamiento del train.csv de Jigsaw (comment_text).

Versión optimizada con reporte técnico JSON:
- Multiprocessing: paraleliza en todos los cores del CPU.
- Regex combinado: un solo patrón para profanidad (1 pasada vs 8).
- Hot-path mínimo: reduce llamadas a funciones Python en el loop interno.
- Reporte: genera JSON con contadores por fase del pipeline.
"""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas y API pública
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = (
    _ROOT
    / "data"
    / "raw"
    / "jigsaw-toxic-comment-classification-challenge"
    / "train"
    / "train.csv"
)

__all__ = [
    "DEFAULT_TRAIN_CSV",
    "PreprocessConfig",
    "preprocess_text",
    "preprocess_text_tracked",
    "load_and_preprocess_train",
]

# ---------------------------------------------------------------------------
# Fase B: patrones wiki / plataforma — UN SOLO REGEX COMBINADO
# ---------------------------------------------------------------------------

_RE_WIKI_ALL = re.compile(
    r"\[\[User:[^\]]*\]\]"
    r"|\(\s*(?:talk|UTC)\s*\)"
    r"|\b\d{1,2}:\d{2},\s*"
    r"(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)"
    r"\s+\d{1,2},\s*\d{4}\b",
    re.IGNORECASE,
)

_RE_URL_IP = re.compile(
    r"https?://\S+|www\.\S+"
    r"|\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d{1,3})\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d{1,3})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Fase C: contracciones
# ---------------------------------------------------------------------------

_CONTRACTION_MAP: dict[str, str] = {
    "shouldn't": "should not", "couldn't": "could not",
    "wouldn't": "would not", "mustn't": "must not",
    "mightn't": "might not", "needn't": "need not",
    "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not",
    "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "can't": "can not",
    "cannot": "can not",
    "could've": "could have", "should've": "should have",
    "would've": "would have", "might've": "might have",
    "must've": "must have", "wouldn't've": "would not have",
    "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is",
    "it's": "it is", "that's": "that is",
    "what's": "what is", "who's": "who is",
    "there's": "there is", "here's": "here is",
    "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have",
    "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would",
    "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will",
    "it'll": "it will", "that'll": "that will",
    "let's": "let us", "y'all": "you all",
}

_RE_CONTRACTIONS = re.compile(
    r"\b("
    + "|".join(
        re.escape(k)
        for k in sorted(_CONTRACTION_MAP, key=len, reverse=True)
    )
    + r")\b"
)

# ---------------------------------------------------------------------------
# Fase C: profanidad — UN SOLO REGEX con grupos nombrados
# ---------------------------------------------------------------------------

_OBF = r"[\*@#.\-]"

_PROFANITY_COMBINED = re.compile(
    rf"(?P<fuck>\bf+{_OBF}*u+{_OBF}*c+{_OBF}*k+(?:ing|er|ed|s)?\b)"
    rf"|(?P<shit>\bs+{_OBF}*h+{_OBF}*i+{_OBF}*t+(?:ting|ty|s)?\b)"
    rf"|(?P<bitch>\bb+{_OBF}*i+{_OBF}*t+{_OBF}*c+{_OBF}*h+(?:es)?\b)"
    rf"|(?P<asshole>\ba+{_OBF}*s+{_OBF}*s+{_OBF}*h+{_OBF}*o+{_OBF}*l+{_OBF}*e+\b)"
    rf"|(?P<cunt>\bc+{_OBF}*u+{_OBF}*n+{_OBF}*t+(?:s)?\b)"
    rf"|(?P<damn>\bd+{_OBF}*a+{_OBF}*m+{_OBF}*n+(?:ed|it)?\b)"
    rf"|(?P<nigger>\bn+{_OBF}*i+{_OBF}*g+{_OBF}*g+{_OBF}*(?:er|a|az|ers)?\b)"
    rf"|(?P<bastard>\bb+{_OBF}*a+{_OBF}*s+{_OBF}*t+{_OBF}*a+{_OBF}*r+{_OBF}*d+(?:s)?\b)",
    re.IGNORECASE,
)


def _profanity_repl(m: re.Match) -> str:
    return m.lastgroup  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fase D–E: limpieza final
# ---------------------------------------------------------------------------

_RE_REPEATED_PUNCT = re.compile(r"([!?.,])\1+")
_RE_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9\s]+")
_RE_WHITESPACE = re.compile(r"\s+")

CAPSLOCK_TOKEN = "CAPSLOCKFLAG"

# ---------------------------------------------------------------------------
# Caracteres especiales a preservar antes de NFKD
# ---------------------------------------------------------------------------

_PRESERVE_MAP: dict[str, str] = {
    "ñ": "<<ENIE>>", "Ñ": "<<ENIE_UPPER>>",
    "ü": "<<UDIER>>", "Ü": "<<UDIER_UPPER>>",
}
_RESTORE_MAP: dict[str, str] = {v: k for k, v in _PRESERVE_MAP.items()}
_RE_PRESERVE = re.compile("|".join(re.escape(c) for c in _PRESERVE_MAP))
_RE_RESTORE = re.compile("|".join(re.escape(t) for t in _RESTORE_MAP))
_CAT_MN = "Mn"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    normalize_unicode: bool = True
    unescape_html: bool = True
    strip_wiki: bool = True
    strip_urls: bool = True
    expand_contractions: bool = True
    deobfuscate: bool = True
    collapse_punct: bool = True
    strip_non_alnum: bool = True
    detect_caps: bool = True
    custom_steps: list[Callable[[str], str]] = field(default_factory=list)


_DEFAULT_CONFIG = PreprocessConfig()

# ---------------------------------------------------------------------------
# Pre-bound methods
# ---------------------------------------------------------------------------

_html_unescape = html.unescape
_nfkd = unicodedata.normalize
_cat = unicodedata.category

_wiki_sub = _RE_WIKI_ALL.sub
_urlip_sub = _RE_URL_IP.sub
_contract_sub = _RE_CONTRACTIONS.sub
_profanity_sub = _PROFANITY_COMBINED.sub
_punct_sub = _RE_REPEATED_PUNCT.sub
_nonalnum_sub = _RE_NON_ALNUM_SPACE.sub
_ws_sub = _RE_WHITESPACE.sub
_preserve_sub = _RE_PRESERVE.sub
_restore_sub = _RE_RESTORE.sub


# ---------------------------------------------------------------------------
# preprocess_text — versión rápida SIN tracking (para multiprocessing)
# ---------------------------------------------------------------------------


def preprocess_text(
    text: str,
    config: PreprocessConfig = _DEFAULT_CONFIG,
) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = str(text)
    if not s.strip():
        return ""

    if config.unescape_html:
        s = _html_unescape(s)
    if config.normalize_unicode:
        s = _preserve_sub(lambda m: _PRESERVE_MAP[m.group()], s)
        s = "".join(c for c in _nfkd("NFKD", s) if _cat(c) != _CAT_MN)
        s = _restore_sub(lambda m: _RESTORE_MAP[m.group()], s)

    caps_flag = False
    if config.detect_caps:
        alpha = upper = 0
        for c in s:
            if c.isalpha():
                alpha += 1
                if c.isupper():
                    upper += 1
        caps_flag = alpha > 0 and (upper / alpha) > 0.5

    if config.strip_wiki:
        s = _wiki_sub(" ", s)
    if config.strip_urls:
        s = _urlip_sub(" ", s)

    s = s.lower()

    if config.expand_contractions:
        s = _contract_sub(lambda m: _CONTRACTION_MAP[m.group()], s)
    if config.deobfuscate:
        s = _profanity_sub(_profanity_repl, s)
    if config.collapse_punct:
        s = _punct_sub(r"\1", s)
    if config.strip_non_alnum:
        s = _nonalnum_sub(" ", s)

    s = _ws_sub(" ", s).strip()

    for step_fn in config.custom_steps:
        s = step_fn(s)
    if caps_flag:
        s = f"{CAPSLOCK_TOKEN} {s}"

    return s


# ---------------------------------------------------------------------------
# preprocess_text_tracked — versión con contadores (single-thread, reporte)
# ---------------------------------------------------------------------------


def preprocess_text_tracked(
    text: str,
    counters: dict[str, int],
    contraction_detail: Counter,
    profanity_detail: Counter,
    config: PreprocessConfig = _DEFAULT_CONFIG,
) -> str:
    """Igual que preprocess_text pero acumula contadores."""
    if text is None or (isinstance(text, float) and text != text):
        counters["empty_or_nan"] += 1
        return ""
    s = str(text)
    if not s.strip():
        counters["empty_or_nan"] += 1
        return ""

    # 1 — HTML unescape
    if config.unescape_html:
        before = s
        s = _html_unescape(s)
        if s != before:
            counters["html_unescaped"] += 1

    # 2 — Unicode NFKD
    if config.normalize_unicode:
        before = s
        s = _preserve_sub(lambda m: _PRESERVE_MAP[m.group()], s)
        s = "".join(c for c in _nfkd("NFKD", s) if _cat(c) != _CAT_MN)
        s = _restore_sub(lambda m: _RESTORE_MAP[m.group()], s)
        if s != before:
            counters["unicode_normalized"] += 1

    # 3 — CAPS
    caps_flag = False
    if config.detect_caps:
        alpha = upper = 0
        for c in s:
            if c.isalpha():
                alpha += 1
                if c.isupper():
                    upper += 1
        caps_flag = alpha > 0 and (upper / alpha) > 0.5
        if caps_flag:
            counters["capslock_detected"] += 1

    # 4 — Wiki noise
    if config.strip_wiki:
        hits = _RE_WIKI_ALL.findall(s)
        if hits:
            counters["wiki_noise_removed"] += 1
            counters["wiki_noise_matches"] += len(hits)
        s = _wiki_sub(" ", s)

    # 5 — URLs/IPs
    if config.strip_urls:
        hits = _RE_URL_IP.findall(s)
        if hits:
            counters["urls_ips_removed"] += 1
            counters["urls_ips_matches"] += len(hits)
        s = _urlip_sub(" ", s)

    # 6 — Lowercase
    s = s.lower()

    # 7 — Contracciones
    if config.expand_contractions:
        found_any = False
        for m in _RE_CONTRACTIONS.finditer(s):
            contraction_detail[m.group()] += 1
            counters["contractions_expanded_total"] += 1
            found_any = True
        s = _contract_sub(lambda m: _CONTRACTION_MAP[m.group()], s)
        if found_any:
            counters["rows_with_contractions"] += 1

    # 8 — Profanidad
    if config.deobfuscate:
        for m in _PROFANITY_COMBINED.finditer(s):
            profanity_detail[m.lastgroup] += 1
            counters["profanity_deobfuscated_total"] += 1
        s = _profanity_sub(_profanity_repl, s)

    # 9 — Puntuación repetida
    if config.collapse_punct:
        hits = _RE_REPEATED_PUNCT.findall(s)
        if hits:
            counters["repeated_punct_collapsed"] += 1
            counters["repeated_punct_matches"] += len(hits)
        s = _punct_sub(r"\1", s)

    # 10 — Non-alnum
    if config.strip_non_alnum:
        hits = _RE_NON_ALNUM_SPACE.findall(s)
        if hits:
            counters["non_alnum_stripped"] += 1
            counters["non_alnum_matches"] += len(hits)
        s = _nonalnum_sub(" ", s)

    # 11 — Whitespace
    s = _ws_sub(" ", s).strip()

    # 12 — Custom
    for step_fn in config.custom_steps:
        s = step_fn(s)

    # 13 — CAPSLOCK
    if caps_flag:
        s = f"{CAPSLOCK_TOKEN} {s}"

    return s


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------

_MP_CONFIG: PreprocessConfig = _DEFAULT_CONFIG


def _worker_init(config: PreprocessConfig) -> None:
    global _MP_CONFIG
    _MP_CONFIG = config


def _process_chunk(texts: list[str]) -> list[str]:
    cfg = _MP_CONFIG
    return [preprocess_text(t, cfg) for t in texts]


# ---------------------------------------------------------------------------
# Generador de reporte JSON
# ---------------------------------------------------------------------------


def _generate_report(
    *,
    input_path: str,
    output_path: str,
    report_path: str,
    total_rows: int,
    n_nan: int,
    counters: dict[str, int],
    contraction_detail: dict[str, int],
    profanity_detail: dict[str, int],
    text_length_stats: dict[str, float],
    elapsed_seconds: float,
    n_workers: int,
    config: PreprocessConfig,
) -> dict:
    return {
        "meta": {
            "script": "preprocessing.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "workers_used": n_workers,
        },
        "files": {
            "input": input_path,
            "output": output_path,
            "report": report_path,
        },
        "dataset": {
            "total_rows": total_rows,
            "nan_filled": n_nan,
            "empty_or_nan_texts": counters.get("empty_or_nan", 0),
        },
        "pipeline_config": {
            "normalize_unicode": config.normalize_unicode,
            "unescape_html": config.unescape_html,
            "strip_wiki": config.strip_wiki,
            "strip_urls": config.strip_urls,
            "expand_contractions": config.expand_contractions,
            "deobfuscate": config.deobfuscate,
            "collapse_punct": config.collapse_punct,
            "strip_non_alnum": config.strip_non_alnum,
            "detect_caps": config.detect_caps,
            "custom_steps_count": len(config.custom_steps),
        },
        "transformations": {
            "html_unescape": {
                "rows_affected": counters.get("html_unescaped", 0),
            },
            "unicode_normalization": {
                "rows_affected": counters.get("unicode_normalized", 0),
            },
            "capslock_detection": {
                "rows_flagged": counters.get("capslock_detected", 0),
            },
            "wiki_noise_removal": {
                "rows_affected": counters.get("wiki_noise_removed", 0),
                "total_matches": counters.get("wiki_noise_matches", 0),
            },
            "url_ip_removal": {
                "rows_affected": counters.get("urls_ips_removed", 0),
                "total_matches": counters.get("urls_ips_matches", 0),
            },
            "contraction_expansion": {
                "rows_affected": counters.get("rows_with_contractions", 0),
                "total_expansions": counters.get("contractions_expanded_total", 0),
                "top_20": dict(
                    Counter(contraction_detail).most_common(20)
                ),
            },
            "profanity_deobfuscation": {
                "total_matches": counters.get("profanity_deobfuscated_total", 0),
                "by_word": dict(
                    Counter(profanity_detail).most_common(20)
                ),
            },
            "repeated_punct_collapse": {
                "rows_affected": counters.get("repeated_punct_collapsed", 0),
                "total_matches": counters.get("repeated_punct_matches", 0),
            },
            "non_alnum_strip": {
                "rows_affected": counters.get("non_alnum_stripped", 0),
                "total_matches": counters.get("non_alnum_matches", 0),
            },
        },
        "text_length_stats_post": text_length_stats,
    }


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------


def load_and_preprocess_train(
    path: Path | str | None = None,
    *,
    text_column: str = "comment_text",
    save_path: Path | str | None = None,
    report_path: Path | str | None = None,
    nrows: int | None = None,
    config: PreprocessConfig = _DEFAULT_CONFIG,
    n_workers: int | None = None,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Lee train.csv y aplica preprocesamiento.

    Si report_path se indica, genera un JSON técnico con estadísticas
    detalladas de cada fase. En ese caso usa single-thread (tracked).
    Sin reporte usa multiprocessing para máxima velocidad.
    """
    import time

    csv_path = Path(path) if path is not None else DEFAULT_TRAIN_CSV
    logger.info("Leyendo %s (nrows=%s)", csv_path, nrows)

    t0 = time.perf_counter()

    df = pd.read_csv(csv_path, nrows=nrows)
    if text_column not in df.columns:
        raise KeyError(f"Columna ausente: {text_column!r}")

    out = df.copy()
    n_nan = int(out[text_column].isna().sum())
    if n_nan:
        logger.info("Filas con NaN en '%s': %d", text_column, n_nan)

    texts = out[text_column].fillna("").tolist()
    n = len(texts)

    workers = n_workers or max(1, cpu_count() - 1)
    generate_report = report_path is not None

    if generate_report:
        # --- MODO TRACKED (single-thread para contadores precisos) ---
        logger.info(
            "Procesando %d filas con tracking (single-thread para reporte)", n
        )
        counters: dict[str, int] = Counter()  # type: ignore[assignment]
        contraction_detail: Counter = Counter()
        profanity_detail: Counter = Counter()

        processed = []
        for t in texts:
            result = preprocess_text_tracked(
                t, counters, contraction_detail, profanity_detail, config
            )
            processed.append(result)

        actual_workers = 1
    else:
        # --- MODO RÁPIDO (multiprocessing, sin tracking) ---
        if workers <= 1 or n < chunk_size:
            logger.info("Procesando %d filas single-thread", n)
            processed = [preprocess_text(t, config) for t in texts]
            actual_workers = 1
        else:
            chunks = [
                texts[i : i + chunk_size] for i in range(0, n, chunk_size)
            ]
            logger.info(
                "Procesando %d filas en %d chunks con %d workers",
                n, len(chunks), workers,
            )
            with Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(config,),
            ) as pool:
                results = pool.map(_process_chunk, chunks)
            processed = []
            for chunk_result in results:
                processed.extend(chunk_result)
            actual_workers = workers

        counters = {}
        contraction_detail = Counter()
        profanity_detail = Counter()

    out[text_column] = processed
    elapsed = time.perf_counter() - t0

    n_caps = sum(1 for t in processed if t.startswith(CAPSLOCK_TOKEN))
    logger.info(
        "Preprocesamiento completado: %d filas, %d CAPSLOCK (%.2fs)",
        len(out), n_caps, elapsed,
    )

    # Guardar CSV
    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        if sp.suffix.lower() == ".parquet":
            out.to_parquet(sp, index=False)
        else:
            out.to_csv(sp, index=False)
        logger.info("Guardado en %s", sp)

    # Guardar reporte JSON
    if generate_report:
        lengths = out[text_column].str.len()
        text_stats = {
            "mean": round(float(lengths.mean()), 2),
            "median": round(float(lengths.median()), 2),
            "std": round(float(lengths.std()), 2),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "p25": round(float(lengths.quantile(0.25)), 2),
            "p75": round(float(lengths.quantile(0.75)), 2),
            "p95": round(float(lengths.quantile(0.95)), 2),
        }

        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)

        report = _generate_report(
            input_path=str(csv_path),
            output_path=str(save_path) if save_path else "",
            report_path=str(rp),
            total_rows=n,
            n_nan=n_nan,
            counters=dict(counters),
            contraction_detail=dict(contraction_detail),
            profanity_detail=dict(profanity_detail),
            text_length_stats=text_stats,
            elapsed_seconds=elapsed,
            n_workers=actual_workers,
            config=config,
        )

        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Reporte técnico guardado en %s", rp)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Preprocesar train.csv de Jigsaw"
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--no-report", action="store_true",
        help="Desactivar reporte JSON (usa multiprocessing rápido)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output = args.output or str(
        _ROOT / "data" / "processed" / "train_clean.csv"
    )
    report = None if args.no_report else str(
        _ROOT / "data" / "processed" / "preprocessing_report.json"
    )

    t0 = time.perf_counter()
    df = load_and_preprocess_train(
        path=args.input,
        save_path=output,
        report_path=report,
        nrows=args.nrows,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nCSV guardado en:    {output}")
    if report:
        print(f"Reporte JSON en:   {report}")
    print(f"Filas: {len(df):,} | Columnas: {list(df.columns)}")
    print(f"Tiempo total: {elapsed:.2f}s")