import csv
from pathlib import Path

def summarize():
    analysis_file = Path("reports/anti-pattern-analysis/private_method_analysis.csv")
    if not analysis_file.exists():
        print("Analysis file not found.")
        return

    pattern_3_candidates = []
    pattern_2_candidates = []
    category_a_top = []

    with open(analysis_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Pattern 3" in row["pattern"]:
                pattern_3_candidates.append(row)
            elif "Pattern 2" in row["pattern"]:
                pattern_2_candidates.append(row)
            elif "Pattern 1" in row["pattern"]:
                category_a_top.append(row)

    print("# Anti-Pattern Analysis Summary")
    print(f"\n## Pattern 3 (Dead Code Candidates): {len(pattern_3_candidates)}")
    print("These are defined in src/ but only called from tests (or not at all). They can likely be deleted.")
    print("| Name | Definition File | Test Usages | Pattern |")
    print("| :--- | :--- | :--- | :--- |")
    for c in sorted(pattern_3_candidates, key=lambda x: int(x["test_usages"]), reverse=True):
        print(f"| {c['name']} | {c['def_file']} | {c['test_usages']} | {c['pattern']} |")

    print(f"\n## Pattern 2 (Test Utilities): {len(pattern_2_candidates)}")
    print("These are defined in tests/ with a leading underscore. They should be renamed to public.")
    print("| Name | Definition File | Test Usages | Pattern |")
    print("| :--- | :--- | :--- | :--- |")
    for c in sorted(pattern_2_candidates, key=lambda x: int(x["test_usages"]), reverse=True)[:20]:
        print(f"| {c['name']} | {c['def_file']} | {c['test_usages']} | {c['pattern']} |")

    category_a_top.sort(key=lambda x: int(x["test_usages"]), reverse=True)
    print(f"\n## Top Category A (Internal Logic) to Fix: {len(category_a_top)} unique methods")
    print("| Name | Definition File | Test Usages | Src Usages |")
    print("| :--- | :--- | :--- | :--- |")
    for c in category_a_top[:20]:
        print(f"| {c['name']} | {c['def_file']} | {c['test_usages']} | {c['src_usages']} |")

if __name__ == "__main__":
    summarize()
