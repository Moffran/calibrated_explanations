import ast
import collections
import csv
import sys
import os
from pathlib import Path

def get_private_definitions(root_path):
    root = str(root_path)
    definitions = {} # name -> {file, line, type}
    skip_dirs = {".ci-env", "venv", ".venv", ".git", "site-packages", "__pycache__", "build", "dist"}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip_dirs
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]


        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            p = Path(dirpath) / filename
            try:
                src = p.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(p))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_") and not node.name.startswith("__") and node.name != "_":
                        definitions[node.name] = {
                            "file": str(p.relative_to(root_path)),
                            "line": node.lineno,
                            "type": "method"
                        }
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith("_") and not target.id.startswith("__") and target.id != "_":
                             definitions[target.id] = {
                                "file": str(p.relative_to(root_path)),
                                "line": node.lineno,
                                "type": "attribute"
                            }
                        elif isinstance(target, ast.Attribute) and target.attr.startswith("_") and not target.attr.startswith("__") and target.attr != "_":
                             definitions[target.attr] = {
                                "file": str(p.relative_to(root_path)),
                                "line": node.lineno,
                                "type": "attribute"
                            }
    return definitions

def get_usages(root_path, private_names):
    root = str(root_path)
    usages = collections.defaultdict(list) # name -> list of {file, line, is_test}
    skip_dirs = {".ci-env", "venv", ".venv", ".git", "site-packages", "__pycache__", "build", "dist"}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip_dirs
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        is_test_dir = "tests" in Path(dirpath).parts

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            p = Path(dirpath) / filename
            is_test = is_test_dir or "test" in filename

            try:
                src = p.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(p))
            except Exception:
                continue

            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    name = node.func.id

                if name and name in private_names:
                    usages[name].append({
                        "file": str(p.relative_to(root_path)),
                        "line": node.lineno,
                        "is_test": is_test
                    })
    return usages

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze private method definitions and usages.")
    parser.add_argument("src_root", default="src", help="Source root directory.")
    parser.add_argument("test_root", default="tests", help="Test root directory.")
    parser.add_argument("--output", default="reports/private_method_analysis.csv", help="Output CSV file.")

    args = parser.parse_args()

    src_root = Path(args.src_root).absolute()
    test_root = Path(args.test_root).absolute()
    project_root = src_root.parent

    print(f"Analyzing library private definitions in {src_root}...")
    lib_definitions = get_private_definitions(src_root)
    for info in lib_definitions.values():
        info["scope"] = "library"
    print(f"Found {len(lib_definitions)} library definitions.")

    print(f"Analyzing test private definitions in {test_root}...")
    test_definitions = get_private_definitions(test_root)
    for info in test_definitions.values():
        info["scope"] = "test"
    print(f"Found {len(test_definitions)} test definitions.")

    # Combine: library definitions take priority for categorization
    all_definitions = {**test_definitions, **lib_definitions}

    print(f"Scanning project root for usages of {len(all_definitions)} private members...")
    usages = get_usages(project_root, set(all_definitions.keys()))

    report = []
    for name, def_info in all_definitions.items():
        use_list = usages.get(name, [])
        src_usages = [u for u in use_list if "src" in Path(u["file"]).parts]
        test_usages = [u for u in use_list if "tests" in Path(u["file"]).parts]

        actual_src_usages = [u for u in src_usages if not (u["file"] == def_info["file"] and u["line"] == def_info["line"])]

        pattern = "Unknown"
        if def_info["scope"] == "library":
            if len(actual_src_usages) == 0 and len(test_usages) == 0:
                pattern = "Pattern 3 (Completely Dead)"
            elif len(actual_src_usages) == 0 and len(test_usages) > 0:
                pattern = "Pattern 3/2 (Only Tests)"
            elif len(actual_src_usages) > 0 and len(test_usages) > 0:
                pattern = "Pattern 1 (Inter-module / Testing leaked)"
            elif len(actual_src_usages) > 0 and len(test_usages) == 0:
                pattern = "Consistent (Internal Only)"
        else:
            # Test defined helper
            if len(test_usages) > 1:
                pattern = "Pattern 2 (Shared Test Utility)"
            else:
                pattern = "Pattern 2 (Local Test Helper)"

        report.append({
            "name": name,
            "def_file": def_info["file"],
            "def_line": def_info["line"],
            "src_usages": len(actual_src_usages),
            "test_usages": len(test_usages),
            "pattern": pattern,
            "scope": def_info["scope"]
        })

    out_file = Path(args.output)
    out_file.parent.mkdir(exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "def_file", "def_line", "src_usages", "test_usages", "pattern", "scope"])
        writer.writeheader()
        writer.writerows(report)

    print(f"Analysis complete. Report written to {out_file}")

if __name__ == "__main__":
    main()
