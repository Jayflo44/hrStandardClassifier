"""Tier 2 — Motor semántico basado en toxic-bert.

Clasifica textos que el Tier 1 no pudo resolver correctamente.
Usa unitary/toxic-bert con umbrales configurables para enrutar
mensajes a OK, FLAGGED o AMBIGUOUS (escalar a Tier 3).
"""

from __future__ import annotations

from transformers import pipeline as hf_pipeline


class Tier2SemanticEngine:
    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        threshold_safe: float = 0.20,
        threshold_toxic: float = 0.80,
        max_length: int = 512,
    ) -> None:
        print(f"Cargando modelo Tier 2: {model_name}...")
        self.classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
        )
        self.threshold_safe = threshold_safe
        self.threshold_toxic = threshold_toxic
        self.max_length = max_length
        self.model_name = model_name
        print("  → Modelo cargado.")

    def get_toxic_prob(self, text: str) -> float:
        """Retorna la probabilidad de toxicidad (0.0 a 1.0)."""
        truncated = text[: self.max_length] if text else ""
        raw_results = self.classifier(truncated)

        # Fix de robustez para diferentes versiones de transformers
        if isinstance(raw_results[0], list):
            results = raw_results[0]
        else:
            results = raw_results

        for item in results:
            if item["label"].lower() == "toxic":
                return float(item["score"])
        return 0.0

    def analyze_and_route(self, text: str) -> dict:
        """Clasifica un texto y retorna decisión + probabilidad + acción."""
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


# --- Prueba rápida ---
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