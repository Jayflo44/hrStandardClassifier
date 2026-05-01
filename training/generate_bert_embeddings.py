from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "hrClassifierSrc" / "data" / "processed" / "train_clean_validated.csv"
TEST_PATH = PROJECT_ROOT / "hrClassifierSrc" / "data" / "processed" / "test_clean_validated.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TRAIN_OUTPUT = MODELS_DIR / "tier2_train_bert_embeddings.npy"
TEST_OUTPUT = MODELS_DIR / "tier2_test_bert_embeddings.npy"
REPORT_OUTPUT = REPORTS_DIR / "bert_embedding_generation_report.json"

TEXT_COL = "comment_text"

BERT_MODEL_NAME = "unitary/toxic-bert"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 8
EMBEDDING_DIM = 768


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_texts(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column: {TEXT_COL}")

    return df[TEXT_COL].fillna("").astype(str)


def load_bert_encoder():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")

    return tokenizer, model, device


def extract_embeddings_to_memmap(
    texts: pd.Series,
    tokenizer,
    model,
    device,
    output_path: Path,
) -> dict:
    text_list = texts.tolist()
    num_rows = len(text_list)

    temp_path = output_path.with_suffix(".dat")
    progress_path = output_path.with_suffix(".progress.json")

    start_row = 0

    if temp_path.exists() and progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_row = int(progress.get("rows_completed", 0))
        print(f"Resuming {output_path.name} from row {start_row:,}")
        embeddings = np.memmap(
            temp_path,
            dtype="float32",
            mode="r+",
            shape=(num_rows, EMBEDDING_DIM),
        )
    else:
        print(f"Starting new embedding file: {output_path.name}")
        embeddings = np.memmap(
            temp_path,
            dtype="float32",
            mode="w+",
            shape=(num_rows, EMBEDDING_DIM),
        )

    for start in tqdm(
        range(start_row, num_rows, BERT_BATCH_SIZE),
        desc=f"Extracting {output_path.name}",
    ):
        end = min(start + BERT_BATCH_SIZE, num_rows)
        batch_texts = text_list[start:end]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")

        embeddings[start:end] = cls_embeddings
        embeddings.flush()

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rows_completed": end,
                    "total_rows": num_rows,
                    "embedding_dim": EMBEDDING_DIM,
                    "updated_at": utc_now(),
                },
                f,
                indent=2,
            )

    final_array = np.asarray(embeddings)
    np.save(output_path, final_array)

    del embeddings

    if temp_path.exists():
        temp_path.unlink()
    if progress_path.exists():
        progress_path.unlink()

    return {
        "output_path": str(output_path),
        "rows": num_rows,
        "embedding_dim": EMBEDDING_DIM,
        "shape": [num_rows, EMBEDDING_DIM],
    }


def main() -> None:
    ensure_dirs()

    print("Loading train/test text...")
    train_texts = load_texts(TRAIN_PATH)
    test_texts = load_texts(TEST_PATH)

    print(f"Train rows: {len(train_texts):,}")
    print(f"Test rows:  {len(test_texts):,}")
    print(f"Batch size: {BERT_BATCH_SIZE}")
    print(f"Max length: {BERT_MAX_LENGTH}")

    tokenizer, model, device = load_bert_encoder()

    report = {
        "timestamp": utc_now(),
        "bert_model_name": BERT_MODEL_NAME,
        "max_length": BERT_MAX_LENGTH,
        "batch_size": BERT_BATCH_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "outputs": {},
    }

    print("\nGenerating train embeddings...")
    report["outputs"]["train"] = extract_embeddings_to_memmap(
        train_texts,
        tokenizer,
        model,
        device,
        TRAIN_OUTPUT,
    )

    print("\nGenerating test embeddings...")
    report["outputs"]["test"] = extract_embeddings_to_memmap(
        test_texts,
        tokenizer,
        model,
        device,
        TEST_OUTPUT,
    )

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Report saved to: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()