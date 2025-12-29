import csv
import json
import os
from datetime import datetime, timedelta

def generate_allowlist(analysis_file, current_allowlist_file):
    if not os.path.exists(analysis_file):
        print(f"Error: {analysis_file} not found.")
        return

    # Load existing allowlist to preserve any manual entries if needed
    # (Though for this task we might want to refresh it)
    existing_allowlist = []
    if os.path.exists(current_allowlist_file):
        try:
            with open(current_allowlist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_allowlist = data.get("allowlist", [])
        except Exception:
            pass

    # Expiry version: v0.11.0
    expiry = "v0.11.0"

    new_entries = []
    seen = set()

    # We need to know which files use these symbols.
    # The category_a_analysis.csv doesn't have the test files, but private_usage_scan.csv does.
    usage_file = "reports/anti-pattern-analysis/private_usage_scan.csv"

    # Load candidates from category_a_analysis.csv
    candidates = set()
    with open(analysis_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["allow_list_candidate"] == "True":
                candidates.add(row["name"])

    if not os.path.exists(usage_file):
        print(f"Error: {usage_file} not found.")
        return

    with open(usage_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if name in candidates:
                entry = {
                    "file": row["file"],
                    "symbol": name,
                    "expiry": expiry,
                    "reason": f"Pattern 1 remediation: {name} is allow-listed due to {row.get('message', 'internal logic testing')}"
                }
                # Use a key to avoid duplicates
                key = (entry["file"], entry["symbol"])
                if key not in seen:
                    new_entries.append(entry)
                    seen.add(key)

    # Sort for stability
    new_entries.sort(key=lambda x: (x["file"], x["symbol"]))

    output = {"allowlist": new_entries}

    with open(current_allowlist_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Updated {current_allowlist_file} with {len(new_entries)} entries.")

if __name__ == "__main__":
    generate_allowlist(
        "reports/anti-pattern-analysis/category_a_analysis.csv",
        ".github/private_member_allowlist.json"
    )
