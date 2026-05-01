from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from transformers import AutoModel, AutoTokenizer


BERT_MODEL_NAME = "unitary/toxic-bert"
BERT_MAX_LENGTH = 128


class Tier2RuntimeEngine:
    def __init__(self, models_dir: str | Path) -> None:
        models_dir = Path(models_dir)

        self.feature_union = joblib.load(models_dir / "tier2_feature_union_final.joblib")
        self.svd = joblib.load(models_dir / "tier2_svd_final.joblib")
        self.model = joblib.load(models_dir / "tier2_logreg_final.joblib")

        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()

    def extract_bert_embedding(self, text: str) -> np.ndarray:
        text = str(text or "").strip()

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.cpu().numpy()

    def predict_proba(self, text: str) -> float:
        text = str(text or "").strip()

        X_tfidf = self.feature_union.transform([text])
        X_dense = self.svd.transform(X_tfidf)
        X_bert = self.extract_bert_embedding(text)

        X_final = hstack(
            [
                X_tfidf,
                csr_matrix(X_dense),
                csr_matrix(X_bert),
            ]
        ).tocsr()

        prob = self.model.predict_proba(X_final)[0, 1]
        return float(prob)

    def analyze_and_route(
        self,
        text: str,
        threshold_safe: float = 0.35,
        threshold_toxic: float = 0.65,
    ) -> dict:
        toxic_prob = self.predict_proba(text)

        if toxic_prob < threshold_safe:
            return {
                "decision": "OK",
                "confidence": round((1 - toxic_prob) * 100, 2),
                "toxic_prob": round(toxic_prob, 6),
                "action": "Message is within standards.",
                "model_used": "LR + live BERT embedding",
            }

        elif toxic_prob >= threshold_toxic:
            return {
                "decision": "FLAGGED",
                "confidence": round(toxic_prob * 100, 2),
                "toxic_prob": round(toxic_prob, 6),
                "action": "Escalate to HR review.",
                "model_used": "LR + live BERT embedding",
            }

        else:
            return {
                "decision": "AMBIGUOUS",
                "confidence": round(toxic_prob * 100, 2),
                "toxic_prob": round(toxic_prob, 6),
                "action": "Needs manual or Tier 3 review.",
                "model_used": "LR + live BERT embedding",
            }