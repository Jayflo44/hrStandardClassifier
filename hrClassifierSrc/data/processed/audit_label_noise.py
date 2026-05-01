from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_CSV = PROJECT_ROOT / "test_clean_validated.csv"
AUDIT_OUTPUT_CSV = PROJECT_ROOT / "label_audit_candidates.csv"
AUDIT_REPORT_JSON = PROJECT_ROOT / "label_audit_report.json"

ID_COL = "id"
TEXT_COL = "comment_text"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BINARY_LABEL_COL = "is_toxic"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SUSPICIOUS_PATTERNS: dict[str, str] = {
    "violent_phrase": r"\bdeath to\b|\bkill\b|\bhang\b|\bshoot\b|\bmurder\b|\brevenge\b",
    "hate_reference": r"\banti[\s\-]?semit\w*\b|\bnazi\w*\b|\bterrorist\w*\b",
    "profanity": r"\bfuck\w*\b|\bshit\w*\b|\bbitch\w*\b|\bcunt\w*\b|\basshole\w*\b|\bbastard\w*\b",
    "insult_phrase": r"\bidiot\b|\bmoron\b|\bstupid\b|\bpiece of trash\b|\bworthless\b|\bscum\b",
    "threatening_phrase": r"\byou will pay\b|\bi will .*kill\b|\bi will .*hurt\b|\bgo die\b",
}

COMPILED_PATTERNS = {
    name: re.compile(pattern, flags=re.IGNORECASE)
    for name, pattern in SUSPICIOUS_PATTERNS.items()
}


def validate_columns(df: pd.DataFrame) -> None:
    required = [ID_COL, TEXT_COL] + LABEL_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = [ID_COL, TEXT_COL] + LABEL_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[BINARY_LABEL_COL] = df[LABEL_COLS].max(axis=1).astype(int)

    return df

def detect_pattern_hits(text: str) -> list[str]:
    hits: list[str] = []
    for name, pattern in COMPILED_PATTERNS.items():
        if pattern.search(text):
            hits.append(name)
    return hits


def build_audit_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        text = row[TEXT_COL]
        binary_label = int(row[BINARY_LABEL_COL])
        hits = detect_pattern_hits(text)

        if not hits:
            continue

        # suspicious case: looks toxic but labeled non-toxic
        if binary_label == 0:
            rows.append(
                {
                    "id": row[ID_COL],
                    "comment_text": text,
                    "is_toxic": binary_label,
                    "matched_rules": ", ".join(hits),
                    "audit_reason": "suspicious_non_toxic_label",
                    "toxic": row["toxic"],
                    "severe_toxic": row["severe_toxic"],
                    "obscene": row["obscene"],
                    "threat": row["threat"],
                    "insult": row["insult"],
                    "identity_hate": row["identity_hate"],
                }
            )

    audit_df = pd.DataFrame(rows)

    if not audit_df.empty:
        audit_df = audit_df.sort_values(
            by=["matched_rules", "id"],
            ascending=[True, True]
        ).reset_index(drop=True)

    return audit_df


def build_report(df: pd.DataFrame, audit_df: pd.DataFrame) -> dict:
    matched_rule_counts: dict[str, int] = {}
    if not audit_df.empty:
        for rule_name in SUSPICIOUS_PATTERNS.keys():
            matched_rule_counts[rule_name] = int(
                audit_df["matched_rules"].str.contains(rule_name, regex=False).sum()
            )
    else:
        matched_rule_counts = {rule_name: 0 for rule_name in SUSPICIOUS_PATTERNS.keys()}

    report = {
        "timestamp": utc_now(),
        "input_file": str(INPUT_CSV),
        "output_file": str(AUDIT_OUTPUT_CSV),
        "rows_scanned": int(len(df)),
        "audit_candidates_found": int(len(audit_df)),
        "matched_rule_counts": matched_rule_counts,
        "patterns_used": SUSPICIOUS_PATTERNS,
    }
    return report


def save_outputs(audit_df: pd.DataFrame, report: dict) -> None:
    AUDIT_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    audit_df.to_csv(AUDIT_OUTPUT_CSV, index=False)

    with open(AUDIT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit candidates saved to: {AUDIT_OUTPUT_CSV}")
    print(f"Audit report saved to: {AUDIT_REPORT_JSON}")
    print(f"Suspicious rows found: {len(audit_df)}")


def main() -> None:
    print("Loading validated dataset...")
    df = load_dataset(INPUT_CSV)

    print("Scanning for likely mislabeled non-toxic rows...")
    audit_df = build_audit_candidates(df)

    report = build_report(df, audit_df)
    save_outputs(audit_df, report)

    print("\nReport summary:")
    print(json.dumps(report, indent=2))

    if not audit_df.empty:
        print("\nSample suspicious rows:")
        print(audit_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()