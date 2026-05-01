from __future__ import annotations

from pathlib import Path
import pandas as pd

from pipeline import HRPipeline

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
INPUT_PATH = APP_DIR / "data" / "processed" / "pipeline_simulated_inputs.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "reports" / "full_pipeline_simulation.csv"

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def normalize_true_label(row) -> int:
    return int(max(int(row[col]) for col in LABEL_COLS))


def map_decision_to_binary(decision: str) -> int:
    return 1 if decision == "FLAGGED" else 0


def main() -> None:
    print(f"INPUT_PATH: {INPUT_PATH}")
    print(f"INPUT EXISTS: {INPUT_PATH.exists()}")
    print(f"OUTPUT_PATH: {OUTPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    pipeline = HRPipeline()
    results = []

    for _, row in df.iterrows():
        text = row["comment_text"]

        result = pipeline.classify(text)

        # --- Actual outputs ---
        actual_route_raw = result["decided_by"]
        actual_decision = result["final_decision"]

        expected_route = str(row.get("expected_route", "")).strip().upper()
        expected_decision = str(row.get("expected_decision", "")).strip().upper()

        if "tier 1" in actual_route_raw.lower():
            actual_route = "TIER1"
        elif "tier 2" in actual_route_raw.lower():
            actual_route = "TIER2"
        else:
            actual_route = "UNKNOWN"

        route_match = int(expected_route == actual_route)
        decision_match = int(expected_decision == actual_decision.upper())

        # --- Evaluation ---
        route_match = int(expected_route in actual_route)
        decision_match = int(expected_decision == actual_decision)

        results.append(
            {
                "id": row.get("id"),
                "comment_text": text,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "actual_route_raw": actual_route_raw,
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "route_match": route_match,
                "decision_match": decision_match,
                "tier1_status": result["tier1"]["status"] if result["tier1"] else None,
                "tier1_reason": result["tier1"].get("reason") if result["tier1"] else None,
                "tier1_trigger": result["tier1"].get("trigger") if result["tier1"] else None,
                "tier2_decision": result["tier2"]["decision"] if result["tier2"] else None,
                "tier2_prob": result["tier2"]["toxic_prob"] if result["tier2"] else None,
                "tier2_confidence_pct": result["tier2"]["confidence_pct"] if result["tier2"] else None,
                "tier2_action": result["tier2"]["action"] if result["tier2"] else None,
                "reason_summary": result["tier2"].get("reason_summary") if result["tier2"] else None,
                "risk_type": result["tier2"].get("risk_type") if result["tier2"] else None,
                "recommended_action": result["tier2"].get("recommended_action") if result["tier2"] else None,
            }
        )
    print("EXPECTED:", expected_route, "| ACTUAL RAW:", actual_route_raw)

    out_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    route_acc = out_df["route_match"].mean()
    decision_acc = out_df["decision_match"].mean()

    print("Simulation complete.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Route Accuracy: {route_acc:.4f}")
    print(f"Decision Accuracy: {decision_acc:.4f}")
    print("\n--- Misclassified Cases ---\n")

    errors = out_df[out_df["decision_match"] == 0]

    print("\n--- All Results ---\n")

    print(out_df[[
    "comment_text",
    "expected_decision",
    "actual_decision",
    "expected_route",
    "actual_route_raw",
    "tier1_reason",
    "tier2_prob",
    "tier2_decision",
    "tier2_confidence_pct",
    "reason_summary",
    "risk_type",
    "recommended_action"
]])

if __name__ == "__main__":
    main()