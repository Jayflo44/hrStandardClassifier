from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DATASET = PROJECT_ROOT / "test_clean_validated1.csv"
AUDIT_CANDIDATES = PROJECT_ROOT / "label_audit_candidates.csv"

OUTPUT_DATASET = PROJECT_ROOT / "test_clean_audited.csv"
OUTPUT_REPORT = PROJECT_ROOT / "label_audit_correction_report.json"

ID_COL = "id"
TEXT_COL = "comment_text"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

HIGH_CONFIDENCE_SINGLE = {
    "violent_phrase",
    "threatening_phrase",
}

HIGH_CONFIDENCE_COMBINED = [
    {"profanity", "insult_phrase"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def parse_rule_set(rule_text: str) -> set[str]:
    if pd.isna(rule_text):
        return set()
    return {part.strip() for part in str(rule_text).split(",") if part.strip()}


def is_high_confidence(rule_set: set[str]) -> bool:
    if any(rule in rule_set for rule in HIGH_CONFIDENCE_SINGLE):
        return True

    for combo in HIGH_CONFIDENCE_COMBINED:
        if combo.issubset(rule_set):
            return True

    return False


def apply_corrections(
    base_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    corrected_df = base_df.copy()

    candidate_rows = 0
    corrected_rows = 0
    corrected_ids: list[str] = []

    audit_df = audit_df.copy()
    audit_df["rule_set"] = audit_df["matched_rules"].apply(parse_rule_set)
    audit_df["high_confidence"] = audit_df["rule_set"].apply(is_high_confidence)

    high_conf_df = audit_df[audit_df["high_confidence"]].copy()
    candidate_rows = int(len(high_conf_df))

    high_conf_ids = set(high_conf_df[ID_COL].astype(str))

    for idx, row in corrected_df.iterrows():
        row_id = str(row[ID_COL])
        if row_id not in high_conf_ids:
            continue

        row_is_toxic = int(row[LABEL_COLS].max())
        if row_is_toxic == 0:
            corrected_df.at[idx, "toxic"] = 1
            corrected_rows += 1
            corrected_ids.append(row_id)

    report = {
        "timestamp": utc_now(),
        "input_dataset": str(INPUT_DATASET),
        "audit_candidates_file": str(AUDIT_CANDIDATES),
        "output_dataset": str(OUTPUT_DATASET),
        "rows_in_input_dataset": int(len(base_df)),
        "audit_rows_loaded": int(len(audit_df)),
        "high_confidence_candidates": candidate_rows,
        "rows_corrected": corrected_rows,
        "correction_policy": {
            "single_rules": sorted(HIGH_CONFIDENCE_SINGLE),
            "combined_rules": [sorted(list(x)) for x in HIGH_CONFIDENCE_COMBINED],
        },
        "corrected_ids_sample": corrected_ids[:100],
        "label_distribution_before": {
            "non_toxic": int((base_df[LABEL_COLS].max(axis=1) == 0).sum()),
            "toxic": int((base_df[LABEL_COLS].max(axis=1) == 1).sum()),
        },
        "label_distribution_after": {
            "non_toxic": int((corrected_df[LABEL_COLS].max(axis=1) == 0).sum()),
            "toxic": int((corrected_df[LABEL_COLS].max(axis=1) == 1).sum()),
        },
    }

    return corrected_df, report


def save_outputs(df: pd.DataFrame, report: dict) -> None:
    df.to_csv(OUTPUT_DATASET, index=False)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved corrected dataset to: {OUTPUT_DATASET}")
    print(f"Saved correction report to: {OUTPUT_REPORT}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


def main() -> None:
    print("Loading base validated dataset...")
    base_df = load_csv(INPUT_DATASET)

    print("Loading audit candidates...")
    audit_df = load_csv(AUDIT_CANDIDATES)

    print("Applying high-confidence corrections...")
    corrected_df, report = apply_corrections(base_df, audit_df)

    print("Saving outputs...")
    save_outputs(corrected_df, report)

    print("\nCorrection summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()