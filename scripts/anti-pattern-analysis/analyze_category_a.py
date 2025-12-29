import csv
import os
from pathlib import Path
import collections

def analyze_category_a(analysis_file, usage_file):
    if not os.path.exists(analysis_file) or not os.path.exists(usage_file):
        print("Error: Analysis or usage files missing. Run analyze_private_methods.py and scan_private_usage.py first.")
        return

    # Load analysis data
    analysis_data = {}
    with open(analysis_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            analysis_data[row["name"]] = row

    # Load usage data
    usages = collections.defaultdict(list)
    with open(usage_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == "Category A: Internal Logic Testing":
                usages[row["name"]].append(row)

    report = []
    for name, use_list in usages.items():
        def_info = analysis_data.get(name, {})
        test_usages = len(use_list)
        src_usages = int(def_info.get("src_usages", 0))
        def_file = def_info.get("def_file", "Unknown")

        # Heuristics for allow-list candidates
        is_name_mangled = name.startswith("_") and "__" in name
        is_factory = name.startswith("_from_") or name.endswith("_from_config")
        has_public_accessor = False

        # Check for common public accessors (heuristic)
        public_name = name.lstrip("_")
        # This is a simple check, could be improved by scanning src for the public name

        allow_list_candidate = False
        reason = ""

        if is_name_mangled:
            allow_list_candidate = True
            reason = "Name-mangled internal state"
        elif is_factory:
            allow_list_candidate = True
            reason = "Internal factory/setup bypass"
        elif test_usages > 20:
            allow_list_candidate = True
            reason = "High usage volume (refactor risk)"
        elif "legacy" in def_file:
            allow_list_candidate = True
            reason = "Legacy component maintenance"

        report.append({
            "name": name,
            "def_file": def_file,
            "test_usages": test_usages,
            "src_usages": src_usages,
            "is_name_mangled": is_name_mangled,
            "allow_list_candidate": allow_list_candidate,
            "reason": reason,
            "remediation_strategy": "Refactor to public API" if not allow_list_candidate else "Add to allow-list (temporary)"
        })

    # Sort by test usages descending
    report.sort(key=lambda x: x["test_usages"], reverse=True)

    output_file = "reports/anti-pattern-analysis/category_a_analysis.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "def_file", "test_usages", "src_usages", "is_name_mangled", "allow_list_candidate", "reason", "remediation_strategy"])
        writer.writeheader()
        writer.writerows(report)

    print(f"Category A analysis complete. Report written to {output_file}")

    # Print summary
    candidates = [r for r in report if r["allow_list_candidate"]]
    print(f"\nFound {len(report)} Category A methods.")
    print(f"Identified {len(candidates)} potential allow-list candidates.")

    print("\nTop Allow-list Candidates:")
    for c in candidates[:10]:
        print(f"- {c['name']} ({c['test_usages']} usages): {c['reason']}")

if __name__ == "__main__":
    analyze_category_a(
        "reports/anti-pattern-analysis/private_method_analysis.csv",
        "reports/anti-pattern-analysis/private_usage_scan.csv"
    )
