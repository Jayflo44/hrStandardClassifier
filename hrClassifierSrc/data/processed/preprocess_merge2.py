from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

TEST_TEXT_PATH = PROJECT_ROOT / "test.csv"
TEST_LABELS_PATH = PROJECT_ROOT / "test_labels.csv"

OUTPUT_CSV = PROJECT_ROOT / "test_clean_validated1.csv"
OUTPUT_REPORT = PROJECT_ROOT / "test_validation_report1.json"

ID_COL = "id"
TEXT_COL = "comment_text"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TEST_TEXT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_TEXT_PATH}")
    if not TEST_LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_LABELS_PATH}")

    test_df = pd.read_csv(TEST_TEXT_PATH)
    labels_df = pd.read_csv(TEST_LABELS_PATH)

    validate_columns(test_df, [ID_COL, TEXT_COL], "test.csv")
    validate_columns(labels_df, [ID_COL] + LABEL_COLS, "test_labels.csv")

    return test_df, labels_df


def merge_inputs(test_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    merged = test_df.merge(labels_df, on=ID_COL, how="inner", validate="one_to_one")
    return merged


def clean_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {
        "timestamp": utc_now(),
        "input_rows": int(len(df)),
        "drops": {},
    }

    # 1. Drop duplicate ids
    dup_id_mask = df.duplicated(subset=[ID_COL], keep="first")
    dup_id_count = int(dup_id_mask.sum())
    if dup_id_count > 0:
        df = df.loc[~dup_id_mask].copy()
    report["drops"]["duplicate_ids"] = dup_id_count

    # 2. Fill missing text, then drop empty text
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    empty_text_mask = df[TEXT_COL].str.strip() == ""
    empty_text_count = int(empty_text_mask.sum())
    if empty_text_count > 0:
        df = df.loc[~empty_text_mask].copy()
    report["drops"]["empty_comment_text"] = empty_text_count

    # 3. Drop rows with missing labels
    missing_label_mask = df[LABEL_COLS].isnull().any(axis=1)
    missing_label_count = int(missing_label_mask.sum())
    if missing_label_count > 0:
        df = df.loc[~missing_label_mask].copy()
    report["drops"]["missing_label_values"] = missing_label_count

    # 4. Drop rows with invalid label values
    # Expected values for Jigsaw test_labels are often 0, 1, or -1
    valid_label_values = {-1, 0, 1}
    invalid_label_mask = ~df[LABEL_COLS].isin(valid_label_values).all(axis=1)
    invalid_label_count = int(invalid_label_mask.sum())
    if invalid_label_count > 0:
        df = df.loc[~invalid_label_mask].copy()
    report["drops"]["invalid_label_values"] = invalid_label_count

    # 5. Drop rows containing any -1 labels (unscored / unusable for evaluation)
    minus_one_mask = (df[LABEL_COLS] == -1).any(axis=1)
    minus_one_count = int(minus_one_mask.sum())
    if minus_one_count > 0:
        df = df.loc[~minus_one_mask].copy()
    report["drops"]["rows_with_minus_one_labels"] = minus_one_count

    # 6. Cast labels to int
    for col in LABEL_COLS:
        df[col] = df[col].astype(int)

    # 7. Create binary label
    df["is_toxic"] = df[LABEL_COLS].max(axis=1).astype(int)

    # 8. Final column order
    final_cols = [ID_COL, TEXT_COL] + LABEL_COLS + ["is_toxic"]
    df = df[final_cols].copy()

    report["output_rows"] = int(len(df))
    report["rows_removed_total"] = int(report["input_rows"] - report["output_rows"])
    report["label_distribution"] = {
        "non_toxic": int((df["is_toxic"] == 0).sum()),
        "toxic": int((df["is_toxic"] == 1).sum()),
    }

    return df, report


def save_outputs(df: pd.DataFrame, report: dict) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved cleaned dataset to: {OUTPUT_CSV}")
    print(f"Saved validation report to: {OUTPUT_REPORT}")
    print(f"Rows in cleaned dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")


def main() -> None:
    print("Loading test.csv and test_labels.csv...")
    test_df, labels_df = load_inputs()

    print("Merging on id...")
    merged_df = merge_inputs(test_df, labels_df)
    print(f"Merged rows: {len(merged_df)}")

    print("Validating and cleaning...")
    clean_df, report = clean_and_validate(merged_df)

    print("Saving outputs...")
    save_outputs(clean_df, report)

    print("\nValidation summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()