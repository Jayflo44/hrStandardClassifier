"""Pipeline orquestador — Tier 1 → Tier 2 → Tier N.

Uso como clasificador en vivo:
  from pipeline import HRPipeline
  p = HRPipeline()
  result = p.classify("you are an idiot")

Uso CLI para evaluación:
  python pipeline.py                     # Pipeline completo (Tier 1 + Tier 2)
  python pipeline.py --tier 1            # Solo evaluación Tier 1
  python pipeline.py --tier 2            # Solo evaluación Tier 2
  python pipeline.py --message "texto"   # Clasificar un mensaje en vivo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tier_one_bouncer import TierOneBouncer

_ROOT = Path(__file__).resolve().parent
RULES_PATH = _ROOT / "hr_rules.yaml"


class HRPipeline:
    """Pipeline de clasificación HR en vivo: Tier 1 → Tier 2."""

    def __init__(self, rules_path: str | Path = RULES_PATH) -> None:
        self.bouncer = TierOneBouncer(rules_path)
        self._tier2 = None  # lazy load — solo se carga si se necesita

    @property
    def tier2(self):
        """Carga el Tier 2 solo cuando se necesita (lazy loading)."""
        if self._tier2 is None:
            from TierTwoSemanticEngine import Tier2SemanticEngine
            self._tier2 = Tier2SemanticEngine()
        return self._tier2

    def classify(self, raw_text: str) -> dict:
        """Clasifica un mensaje a través del pipeline completo.

        Flujo:
          1. Tier 1 (reglas) → si FLAGGED, retorna inmediatamente.
          2. Si PASS en Tier 1 → Tier 2 (toxic-bert) para análisis semántico.
          3. Tier 2 retorna OK, FLAGGED, o AMBIGUOUS (para futuro Tier 3).
        """
        clean_text = str(raw_text).lower()

        # --- Tier 1: Rule-based ---
        tier1_result = self.bouncer.inspect(clean_text)

        if tier1_result["status"] == "FLAGGED":
            return {
                "final_decision": "FLAGGED",
                "decided_by": "Tier 1 (Rules)",
                "tier1": {
                    "status": "FLAGGED",
                    "reason": tier1_result["reason"],
                    "trigger": tier1_result.get("trigger", ""),
                },
                "tier2": None,
            }

        # --- Tier 2: Semantic analysis ---
        tier2_result = self.tier2.analyze_and_route(clean_text)

        return {
            "final_decision": tier2_result["decision"],
            "decided_by": "Tier 2 (toxic-bert)",
            "tier1": {
                "status": "PASS",
                "reason": None,
                "trigger": None,
            },
            "tier2": {
                "decision": tier2_result["decision"],
                "toxic_prob": tier2_result["toxic_prob"],
                "confidence_pct": tier2_result["confidence"],
                "action": tier2_result["action"],
            },
        }


# ---------------------------------------------------------------------------
# Funciones de evaluación (delegan a los test scripts)
# ---------------------------------------------------------------------------


def _run_tier1_eval() -> None:
    print("\n" + "=" * 70)
    print("  TIER 1 — Rule-Based Bouncer Evaluation")
    print("=" * 70 + "\n")

    from test_rules import run_evaluation
    run_evaluation()


def _run_tier2_eval() -> None:
    print("\n" + "=" * 70)
    print("  TIER 2 — Semantic Engine Evaluation (toxic-bert)")
    print("=" * 70 + "\n")

    from test_tier2 import run_tier2_evaluation
    run_tier2_evaluation()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HR Standards Classifier Pipeline: Tier 1 → Tier 2 → Tier N"
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Ejecutar solo un tier específico (1 o 2)",
    )
    parser.add_argument(
        "--message", type=str, default=None,
        help="Clasificar un mensaje individual en vivo",
    )
    args = parser.parse_args()

    # --- Modo: clasificar un mensaje ---
    if args.message:
        pipeline = HRPipeline()
        result = pipeline.classify(args.message)

        print(f"\nMensaje: '{args.message}'")
        print(f"Decisión final: {result['final_decision']}")
        print(f"Decidido por:   {result['decided_by']}")

        if result["tier1"]["status"] == "FLAGGED":
            print(f"  Tier 1 razón: {result['tier1']['reason']}")
            print(f"  Tier 1 trigger: {result['tier1']['trigger']}")
        elif result["tier2"]:
            print(f"  Tier 2 prob:   {result['tier2']['confidence_pct']}%")
            print(f"  Tier 2 acción: {result['tier2']['action']}")
        return

    # --- Modo: evaluación ---
    t0 = time.perf_counter()

    if args.tier == 1:
        _run_tier1_eval()

    elif args.tier == 2:
        _run_tier2_eval()

    else:
        # Pipeline completo
        _run_tier1_eval()
        _run_tier2_eval()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETO — Tiempo total: {elapsed:.2f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()