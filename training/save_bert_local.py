from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
BERT_LOCAL_DIR = MODELS_DIR / "toxic_bert_local"

MODEL_NAME = "unitary/toxic-bert"

BERT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(BERT_LOCAL_DIR)
model.save_pretrained(BERT_LOCAL_DIR)

print(f"Saved BERT locally to: {BERT_LOCAL_DIR}")