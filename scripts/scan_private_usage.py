import ast
import collections
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_workspace(root_path):
    root = Path(root_path)
    # We only care about tests accessing private members
    # Use a safer iteration to avoid permission errors in venvs
    test_files = []
    for pattern in ["tests/**/*.py", "*test*.py"]:
        try:
            for p in root.rglob(pattern):
                if "site-packages" in str(p) or "venv" in str(p) or ".git" in str(p):
                    continue
                test_files.append(p)
        except OSError:
            continue

    occurrences = []

    for p in test_files:
        if "site-packages" in str(p) or "venv" in str(p):
            continue

        try:
            src = p.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", p, exc)
            continue

        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # obj._private
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    # Exclude dunder methods like __init__, __call__
                    occurrences.append(
                        {
                            "file": str(p.relative_to(root)),
                            "line": node.lineno,
                            "name": node.attr,
                            "type": "attribute",
                        }
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.startswith("_")
                and not node.func.id.startswith("__")
            ):
                # func()
                occurrences.append(
                    {
                        "file": str(p.relative_to(root)),
                        "line": node.lineno,
                        "name": node.func.id,
                        "type": "function_call",
                    }
                )

    return occurrences


def main():
    root = "."
    if len(sys.argv) > 1:
        root = sys.argv[1]

    print(f"Scanning {root} for private member usage in tests...")
    data = scan_workspace(root)

    # Group by name
    counts = collections.Counter(d["name"] for d in data)

    print(f"\nFound {len(data)} occurrences.")
    print("Top 20 most frequent private members accessed:")
    print("-" * 50)
    for name, count in counts.most_common(20):
        print(f"{name:<40} {count}")

    # Write detailed CSV
    out_file = "reports/private_usage_scan.csv"
    Path("reports").mkdir(exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "name", "type"])
        writer.writeheader()
        writer.writerows(data)

    print(f"\nDetailed report written to {out_file}")


if __name__ == "__main__":
    main()
