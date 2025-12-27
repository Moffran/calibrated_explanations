import ast
import collections
import csv
import sys
import os
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
        
        if "src" in Path(def_file).parts:
            if data["pattern"] == "Pattern 3 (Dead Code Candidate)":
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
        # If not in analysis_data (which only scans src/), it's likely defined in tests/
        # but let's be careful. It might be an external lib or dynamic.
        if "_fixtures" in usage_file or "conftest" in usage_file:
             category = "Category B: Test Utilities"
             pattern = "Pattern 2 (Test Utility Fix)"
        else:
            message = "Could not find definition in src/. Might be defined in tests/ or dynamic."
            # We can't be sure without more analysis, but usually private methods 
            # used in tests that aren't in src/ are test helpers.
            category = "Category B: Test Utilities (Likely)"
            pattern = "Pattern 2 (Test Utility Fix)"
        
    return category, pattern, message


def scan_workspace(root_path, analysis_data):
    root = Path(root_path)
    skip_dirs = {".ci-env", "venv", ".venv", ".git", "site-packages", "__pycache__", "build", "dist"}
    
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

def main():
    root = "."
    if len(sys.argv) > 1:
        root = sys.argv[1]

    analysis_file = os.path.join(root, "reports", "private_method_analysis.csv")
    analysis_data = load_analysis(analysis_file)

    print(f"Scanning {root} for private member usage in tests...")
    data = scan_workspace(root, analysis_data)

    print(f"\nFound {len(data)} occurrences.")
    
    # Write detailed CSV
    out_file = os.path.join("reports", "private_usage_scan.csv")
    Path("reports").mkdir(exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "name", "type", "category", "pattern", "message"])
        writer.writeheader()
        writer.writerows(data)

    print(f"\nDetailed report written to {out_file}")
    
    # Summary by Category
    cat_counts = collections.Counter(d["category"] for d in data)
    print("\nUsages by Category:")
    for cat, count in cat_counts.most_common():
        print(f"{cat:<40} {count}")

    # Summary by Pattern
    pat_counts = collections.Counter(d["pattern"] for d in data)
    print("\nUsages by Pattern:")
    for pat, count in pat_counts.most_common():
        print(f"{pat:<40} {count}")

if __name__ == "__main__":
    main()
