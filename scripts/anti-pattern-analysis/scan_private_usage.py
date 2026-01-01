import ast
import collections
import csv
import sys
import os
import json
from pathlib import Path

def load_analysis(analysis_file):
    analysis_data = {}
    if not os.path.exists(analysis_file):
        return analysis_data

    try:
        with open(analysis_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                analysis_data[row["name"]] = row
    except Exception:
        pass
    return analysis_data

def get_category_and_pattern(name, usage_file, analysis_data):
    # Default values
    category = "Unknown"
    pattern = "Unknown"
    message = ""

    # Heuristic for Category D: Factory/Setup Bypass
    if name.startswith("_from_config") or name.endswith("_from_config"):
        category = "Category D: Factory/Setup Bypass"
        pattern = "Pattern 1 (Internal Logic Fix)"
        return category, pattern, message

    # Use analysis data to find where it's defined
    if name in analysis_data:
        data = analysis_data[name]
        def_file = data["def_file"]

        if data.get("scope") == "library":
            if "Pattern 3" in data["pattern"]:
                category = "Category A/C candidate"
                pattern = "Pattern 3 (Dead Code Fix)"
                message = "Defined in src/, only called from tests."
            else:
                category = "Category A: Internal Logic Testing"
                pattern = "Pattern 1 (Internal Logic Fix)"
                message = f"Defined in {def_file}"
        else:
            # Defined outside src/, likely in tests/ or conftest.py
            category = "Category B: Test Utilities"
            pattern = "Pattern 2 (Test Utility Fix)"
            message = f"Defined in {def_file}"
    else:
        # If not in analysis_data
        if "_fixtures" in usage_file or "conftest" in usage_file:
             category = "Category B: Test Utilities"
             pattern = "Pattern 2 (Test Utility Fix)"
        else:
            message = "Could not find definition in analysis data."
            # We can't be sure without more analysis, but usually private methods
            # used in tests that aren't in src/ are test helpers.
            category = "Category B: Test Utilities (Likely)"
            pattern = "Pattern 2 (Test Utility Fix)"

    return category, pattern, message


def scan_workspace(root_path, analysis_data):
    root = Path(root_path)
    skip_dirs = {".ci-env", "venv", ".venv", ".git", "site-packages", "__pycache__", "build", "dist", "scripts"}

    test_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            if filename.endswith(".py") and ("tests" in Path(dirpath).parts or "test" in filename):
                test_files.append(Path(dirpath) / filename)

    occurrences = []

    for p in test_files:
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(p))
        except Exception:
            continue

        rel_path = str(p.relative_to(root))

        for node in ast.walk(tree):
            name = None
            type_str = ""
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") and not node.attr.startswith("__") and node.attr != "_":
                    name = node.attr
                    type_str = "attribute"
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.startswith("_")
                and not node.func.id.startswith("__")
                and node.func.id != "_"
            ):
                name = node.func.id
                type_str = "function_call"

            if name:
                cat, pat, msg = get_category_and_pattern(name, rel_path, analysis_data)
                occurrences.append(
                    {
                        "file": rel_path,
                        "line": node.lineno,
                        "name": name,
                        "type": type_str,
                        "category": cat,
                        "pattern": pat,
                        "message": msg
                    }
                )

    return occurrences

def load_allowlist(allowlist_file):
    if not os.path.exists(allowlist_file):
        return []
    try:
        with open(allowlist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("allowlist", [])
    except Exception:
        return []

def is_allowed(occurrence, allowlist):
    # Normalize occurrence file path to use forward slashes for comparison
    occ_file = occurrence["file"].replace("\\", "/")
    for entry in allowlist:
        # Normalize entry file path
        entry_file = entry["file"].replace("\\", "/")
        if occ_file == entry_file and occurrence["name"] == entry["symbol"]:
            return True
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scan for private member usage in tests.")
    parser.add_argument("roots", nargs="*", default=["."], help="Root directories to scan.")
    parser.add_argument("--output", default="reports/anti-pattern-analysis/private_usage_scan.csv", help="Output CSV file.")
    parser.add_argument("--analysis", default="reports/anti-pattern-analysis/private_method_analysis.csv", help="Path to private_method_analysis.csv.")
    parser.add_argument("--allowlist", default=".github/private_member_allowlist.json", help="Path to private_member_allowlist.json.")
    parser.add_argument("--check", action="store_true", help="Exit with code 1 if non-allowlisted violations are found.")

    args = parser.parse_args()

    analysis_file = args.analysis
    analysis_data = load_analysis(analysis_file)
    allowlist = load_allowlist(args.allowlist)

    all_data = []
    for root in args.roots:
        print(f"Scanning {root} for private member usage in tests...")
        data = scan_workspace(root, analysis_data)
        all_data.extend(data)

    # Filter allowlisted items
    violations = []
    allowed_count = 0
    for d in all_data:
        if is_allowed(d, allowlist):
            allowed_count += 1
        else:
            violations.append(d)

    print(f"\nFound {len(all_data)} total occurrences.")
    print(f"Allowed (in allowlist): {allowed_count}")
    print(f"Violations (not in allowlist): {len(violations)}")

    # Write detailed CSV
    out_file = args.output
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "name", "type", "category", "pattern", "message"])
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\nDetailed report written to {out_file}")

    if args.check:
        if len(violations) > 0:
            print("\nERROR: Found non-allowlisted private member usages in tests:")
            for v in violations:
                print(f"  {v['file']}:{v['line']} - {v['name']} ({v['category']})")
            sys.exit(1)
        else:
            print("\nSUCCESS: No non-allowlisted private member usages found.")

    # Summary by Category
    cat_counts = collections.Counter(d["category"] for d in all_data)
    print("\nUsages by Category:")
    for cat, count in cat_counts.most_common():
        print(f"{cat:<40} {count}")

    # Summary by Pattern
    pat_counts = collections.Counter(d["pattern"] for d in all_data)
    print("\nUsages by Pattern:")
    for pat, count in pat_counts.most_common():
        print(f"{pat:<40} {count}")

if __name__ == "__main__":
    main()
