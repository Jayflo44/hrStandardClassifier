from tier_one_bouncer import TierOneBouncer

bouncer = TierOneBouncer()


def process_message(raw_text):
    clean_text = str(raw_text).lower()

    tier1_result = bouncer.inspect(clean_text)

    if tier1_result["status"] == "FLAGGED":
        return "Flagged", tier1_result["reason"]
    return "Sent to Tier 2", "N/A"


if __name__ == "__main__":
    # Tests rápidos
    tests = [
        "This code is fucking brilliant",
        "You are a great person",
        "My SSN is 123-45-6789",
        "f@ck this",
        "normal professional message",
    ]
    for t in tests:
        result, reason = process_message(t)
        print(f"[{result:15s}] {reason:40s} | {t}")