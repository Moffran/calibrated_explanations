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
        
        if "src" not in Path(dirpath).parts:
            continue
            
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
    root = Path(".").absolute()
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).absolute()
        
    print(f"Analyzing private methods in {root}...")
    
    definitions = get_private_definitions(root)
    print(f"Found {len(definitions)} private definitions in src/")
    
    usages = get_usages(root, set(definitions.keys()))
    
    report = []
    for name, def_info in definitions.items():
        use_list = usages.get(name, [])
        src_usages = [u for u in use_list if not u["is_test"]]
        test_usages = [u for u in use_list if u["is_test"]]
        
        actual_src_usages = [u for u in src_usages if not (u["file"] == def_info["file"] and u["line"] == def_info["line"])]
        
        pattern = "Unknown"
        if len(test_usages) > 0 and len(actual_src_usages) == 0:
            pattern = "Pattern 3 (Dead Code Candidate)"
        elif len(test_usages) > 0:
            pattern = "Pattern 1 (Internal Logic Fix)"
            
        report.append({
            "name": name,
            "def_file": def_info["file"],
            "def_line": def_info["line"],
            "src_usages": len(actual_src_usages),
            "test_usages": len(test_usages),
            "pattern": pattern
        })
        
    out_file = root / "reports" / "private_method_analysis.csv"
    (root / "reports").mkdir(exist_ok=True)
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "def_file", "def_line", "src_usages", "test_usages", "pattern"])
        writer.writeheader()
        writer.writerows(report)
        
    print(f"Analysis complete. Report written to {out_file}")

if __name__ == "__main__":
    main()
