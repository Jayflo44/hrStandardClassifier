from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "artifacts" / "models" / "tier2_finetuned"


class Tier2SemanticEngine:
    """
    Tier 2 semantic engine using Toxic-BERT.

    Designed for:
    - live inference in the HR pipeline
    - future fine-tuning
    - saving/loading trained checkpoints
    """

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        threshold_safe: float = 0.20,
        threshold_toxic: float = 0.80,
        max_length: int = 512,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    ) -> None:
        self.model_name = model_name
        self.threshold_safe = threshold_safe
        self.threshold_toxic = threshold_toxic
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando Tier 2 model en: {self.device}")

        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

        if checkpoint is not None and checkpoint.exists():
            load_source = checkpoint
            print(f"  → Usando checkpoint local: {checkpoint}")
        else:
            load_source = model_name
            if checkpoint_path is not None:
                print(
                    f"  → Checkpoint no encontrado en '{checkpoint_path}'. "
                    f"Usando modelo base: {model_name}"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(load_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_source)
        self.model.to(self.device)
        self.model.eval()

        print(f"  → Modelo cargado desde: {load_source}")

    # ---------------------------------------------------------
    # Core inference
    # ---------------------------------------------------------
    def _prepare_inputs(self, text: str) -> dict:
        text = str(text or "").strip()
        if not text:
            text = "[empty input]"

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict_scores(self, text: str) -> dict:
        """
        Returns raw probabilities for the model labels.
        """
        inputs = self._prepare_inputs(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        id2label = self.model.config.id2label
        scores = {
            str(id2label[i]).lower(): float(probs[i].item())
            for i in range(len(probs))
        }

        return scores

    def get_toxic_prob(self, text: str) -> float:
        """
        Returns the probability for the toxic class.
        """
        scores = self.predict_scores(text)

        toxic_aliases = {"toxic", "label_1", "1"}
        safe_aliases = {"non-toxic", "non_toxic", "clean", "neutral", "label_0", "0"}

        for key, value in scores.items():
            if key.lower() in toxic_aliases:
                return value

        if len(scores) == 2:
            for key, value in scores.items():
                if key.lower() in safe_aliases:
                    return 1.0 - value

        return max(scores.values()) if scores else 0.0

    def analyze_and_route(self, text: str) -> dict:
        """
        Routes text into OK / FLAGGED / AMBIGUOUS
        based on Toxic-BERT output probability.
        """
        toxic_prob = self.get_toxic_prob(text)

        if toxic_prob < self.threshold_safe:
            return {
                "decision": "OK",
                "confidence": round(toxic_prob * 100, 2),
                "toxic_prob": round(toxic_prob, 6),
                "action": "Descartar mensaje. Es seguro.",
            }

        elif toxic_prob > self.threshold_toxic:
            return {
                "decision": "FLAGGED",
                "confidence": round(toxic_prob * 100, 2),
                "toxic_prob": round(toxic_prob, 6),
                "action": "Enviar a panel de RRHH. Violación de política.",
            }

        else:
            return {
                "decision": "AMBIGUOUS",
                "confidence": round(toxic_prob * 100, 2),
                "toxic_prob": round(toxic_prob, 6),
                "action": "Escalar al Tier 3 (LLM) para resolución de contexto profundo.",
            }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Convenience helper for batch routing during testing/backtesting.
        """
        return [self.analyze_and_route(text) for text in texts]

    # ---------------------------------------------------------
    # Fine-tuning readiness
    # ---------------------------------------------------------
    def set_train_mode(self) -> None:
        """
        Enables training mode for fine-tuning.
        """
        self.model.train()

    def set_eval_mode(self) -> None:
        """
        Enables evaluation mode for inference.
        """
        self.model.eval()

    def freeze_encoder(self) -> None:
        """
        Optional: freeze transformer encoder, keep classifier head trainable.
        Useful for lightweight fine-tuning.
        """
        base_model = self.model.base_model
        for param in base_model.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """
        Unfreeze the full model for end-to-end fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    # ---------------------------------------------------------
    # Save / load utilities
    # ---------------------------------------------------------
    def save_checkpoint(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        print(f"Checkpoint guardado en: {save_dir}")


# ---------------------------------------------------------
# Quick test
# ---------------------------------------------------------
if __name__ == "__main__":
    tier2 = Tier2SemanticEngine()

    mensajes_prueba = [
        "Please review the attached quarterly report by tomorrow.",
        "If you deploy that garbage code again, I will break your keyboard.",
        "That presentation was absolute fire! You killed it!",
        "You are a worthless piece of trash.",
        "Great job on the project, everyone contributed well.",
    ]

    for msg in mensajes_prueba:
        resultado = tier2.analyze_and_route(msg)
        print(f"\n[{resultado['decision']:9s}] {resultado['confidence']:6.2f}% | {msg[:80]}")