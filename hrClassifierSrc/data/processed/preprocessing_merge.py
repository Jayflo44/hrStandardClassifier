from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

TEST_TEXT_PATH = BASE_DIR / "test.csv"
TEST_LABELS_PATH = BASE_DIR / "test_labels.csv"

OUTPUT_PATH = BASE_DIR / "test_clean_validated.csv"

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

FINAL_COL_ORDER = ["id", "comment_text"] + LABEL_COLS


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TEST_TEXT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_TEXT_PATH}")

    if not TEST_LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_LABELS_PATH}")

    test_df = pd.read_csv(TEST_TEXT_PATH)
    labels_df = pd.read_csv(TEST_LABELS_PATH)

    validate_columns(test_df, ["id", "comment_text"], "test.csv")
    validate_columns(labels_df, ["id"] + LABEL_COLS, "test_labels.csv")

    return test_df, labels_df


def merge_test_files(test_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = test_df.merge(labels_df, on="id", how="inner", validate="one_to_one")

    if len(merged_df) != len(test_df):
        raise ValueError(
            "Merged test set row count does not match test.csv. "
            "Some ids may be missing or duplicated."
        )

    merged_df["comment_text"] = merged_df["comment_text"].fillna("").astype(str)

    for col in LABEL_COLS:
        merged_df[col] = merged_df[col].astype(int)

    merged_df = merged_df[FINAL_COL_ORDER]
    return merged_df


def save_output(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Merged test file saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

def main() -> None:
    print("Loading test.csv and test_labels.csv...")
    test_df, labels_df = load_inputs()

    print("Merging files on 'id'...")
    merged_df = merge_test_files(test_df, labels_df)

    print("Saving merged test dataset...")
    save_output(merged_df)

    print("\nPreview:")
    print(merged_df.head())


if __name__ == "__main__":
    main()