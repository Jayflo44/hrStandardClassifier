"""
Pipeline orchestrator — Tier 1 → Tier 2 → Tier N.

Live usage:
    from pipeline import HRPipeline
    p = HRPipeline()
    result = p.classify("you are an idiot")

CLI usage:
    python pipeline.py                     # Full pipeline eval (Tier 1 + Tier 2)
    python pipeline.py --tier 1            # Only Tier 1 evaluation
    python pipeline.py --tier 2            # Only Tier 2 evaluation
    python pipeline.py --message "text"    # Classify one live message
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from tier_one_bouncer import TierOneBouncer

_ROOT = Path(__file__).resolve().parent
RULES_PATH = _ROOT / "hr_rules.yaml"
TIER2_CHECKPOINT = _ROOT / "artifacts" / "models" / "tier2_finetuned"

class HRPipeline:
    """HR classification pipeline: Tier 1 → Tier 2."""

    def __init__(self, rules_path: str | Path = RULES_PATH) -> None:
        self.rules_path = Path(rules_path)
        self.bouncer = TierOneBouncer(self.rules_path)
        self._tier2 = None  # lazy-loaded

    @property
    def tier2(self):
        """
        Lazy load Tier 2 only when needed.

        This avoids loading the semantic model during startup if Tier 1
        already catches the message.
        """
        if self._tier2 is None:
            try:
                # Change this import only if your module filename changes.
                from embedd import Tier2SemanticEngine
            except ImportError as e:
                raise ImportError(
                    "Could not import Tier2SemanticEngine from embedd.py. "
                    "Make sure embedd.py exists and all Tier 2 dependencies "
                    "are installed (e.g. transformers, torch)."
                ) from e

            self._tier2 = Tier2SemanticEngine(checkpoint_path=TIER2_CHECKPOINT)
        return self._tier2

    def classify(self, raw_text: str) -> dict[str, Any]:
        """
        Classify a message through the full pipeline.

        Flow:
          1. Tier 1 (rules) → if FLAGGED, return immediately.
          2. If PASS in Tier 1 → Tier 2 semantic analysis.
          3. Tier 2 returns OK, FLAGGED, or AMBIGUOUS.
        """
        clean_text = self._normalize_text(raw_text)

        if not clean_text:
            return {
                "final_decision": "OK",
                "decided_by": "Input Validation",
                "tier1": {
                    "status": "PASS",
                    "reason": None,
                    "trigger": None,
                },
                "tier2": {
                    "decision": "OK",
                    "toxic_prob": 0.0,
                    "confidence_pct": 0.0,
                    "action": "Empty input; nothing to classify.",
                },
            }

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
            "decided_by": "Tier 2 (Semantic Engine)",
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

    @staticmethod
    def _normalize_text(raw_text: str) -> str:
        """Normalize incoming text safely for Tier 1 and Tier 2."""
        return str(raw_text or "").strip().lower()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_tier1_eval() -> None:
    print("\n" + "=" * 70)
    print("  TIER 1 — Rule-Based Bouncer Evaluation")
    print("=" * 70 + "\n")

    try:
        from test_rules import run_evaluation
    except ImportError as e:
        raise ImportError(
            "Could not import run_evaluation from test_rules.py."
        ) from e

    run_evaluation()


def _run_tier2_eval() -> None:
    print("\n" + "=" * 70)
    print("  TIER 2 — Semantic Engine Evaluation")
    print("=" * 70 + "\n")

    try:
        from test_tier2 import run_tier2_evaluation
    except ImportError as e:
        raise ImportError(
            "Could not import run_tier2_evaluation from test_tier2.py."
        ) from e

    run_tier2_evaluation()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HR Standards Classifier Pipeline: Tier 1 → Tier 2 → Tier N"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only one tier evaluation (1 or 2).",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Classify a single live message.",
    )
    args = parser.parse_args()

    # --- Mode: classify one message ---
    if args.message is not None:
        pipeline = HRPipeline()
        result = pipeline.classify(args.message)

        print(f"\nMessage: '{args.message}'")
        print(f"Final decision: {result['final_decision']}")
        print(f"Decided by:     {result['decided_by']}")

        if result["tier1"]["status"] == "FLAGGED":
            print(f"  Tier 1 reason:  {result['tier1']['reason']}")
            print(f"  Tier 1 trigger: {result['tier1']['trigger']}")
        elif result["tier2"] is not None:
            print(f"  Tier 2 prob:    {result['tier2']['confidence_pct']}%")
            print(f"  Tier 2 action:  {result['tier2']['action']}")
        return

    # --- Mode: evaluation ---
    t0 = time.perf_counter()

    if args.tier == 1:
        _run_tier1_eval()
    elif args.tier == 2:
        _run_tier2_eval()
    else:
        _run_tier1_eval()
        _run_tier2_eval()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  FULL PIPELINE — Total time: {elapsed:.2f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()