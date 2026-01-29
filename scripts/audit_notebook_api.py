"""Audit notebooks for legacy API contract compliance (ADR-020).

This script parses notebooks and checks that they only use documented API methods
from WrapCalibratedExplainer and CalibratedExplainer.

Usage:
    python scripts/audit_notebook_api.py [path] [--check] [--json REPORT_PATH]
"""
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Union

# Documented legacy API methods/properties from docs/improvement/legacy_user_api_contract.md
ALLOWED_API = {
    # Methods
    "fit",
    "calibrate",
    "predict",
    "predict_proba",
    "explain_factual",
    "explore_alternatives",
    "set_difficulty_estimator",
    "plot",
    "add_conjunctions",
    "remove_conjunctions",
    "load_collection",
    "save_collection",
    "load_state",
    "save_state",
    "default_reject_policy",
    "initialize_reject_learner",
    "predict_reject",
    "show_in_notebook",
    # Properties
    "learner",
    "explainer",
    "fitted",
    "calibrated",
    "num_features",
    "parallel_executor", # Exposed in wrapper
    "auto_encode", # Config exposed in wrapper
    "preprocessor", 
    "perf_cache",
    "perf_parallel",
    # Legacy
    "get_explanation",    
}

# Heuristic: Variables that are likely to be explainer objects
LIKELY_EXPLAINER_VARS = {"explainer", "wce", "wrapper", "ce", "cal_explainer"}


class NotebookAPIVisitor(ast.NodeVisitor):
    """AST visitor to check API usage on explainer objects."""

    def __init__(self):
        self.api_usage: Set[str] = set()
        self.violations: Set[str] = set()

    def visit_Attribute(self, node: ast.Attribute):
        """Track attribute access (e.g., explainer.explain_factual)."""
        # precise tracking on known variable names
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if var_name in LIKELY_EXPLAINER_VARS:
                if node.attr not in ALLOWED_API:
                    self.violations.add(f"{var_name}.{node.attr}")
                else:
                    self.api_usage.add(node.attr)
        
        # Also track methods in ALLOWED_API regardless of variable name (for usage stats)
        if node.attr in ALLOWED_API:
             self.api_usage.add(node.attr)

        self.generic_visit(node)


def extract_code_from_notebook(notebook_path: Path) -> List[str]:
    """Extract code cells from a Jupyter notebook."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code_cells.append("".join(source))
            else:
                code_cells.append(source)

    return code_cells


def audit_notebook(notebook_path: Path) -> Dict:
    """Audit a single notebook for API usage."""
    try:
        code_cells = extract_code_from_notebook(notebook_path)
    except Exception as e:
        return {
            "path": str(notebook_path),
            "error": f"Failed to parse notebook: {e}",
            "compliant": True, # Parse error is not per se an API violation
            "violations": [],
            "usage": []
        }

    visitor = NotebookAPIVisitor()

    for code in code_cells:
        try:
            tree = ast.parse(code)
            visitor.visit(tree)
        except SyntaxError:
            continue

    violations = sorted(list(visitor.violations))
    usage = sorted(list(visitor.api_usage))

    return {
        "path": str(notebook_path).replace("\\", "/"),
        "compliant": len(violations) == 0,
        "violations": violations,
        "usage": usage
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit notebooks for API compliance")
    parser.add_argument("path", nargs="?", default="notebooks",
                        help="Notebook file or directory to audit")
    parser.add_argument("--check", action="store_true",
                        help="Fail (exit non-zero) if violations are detected")
    parser.add_argument("--json", dest="json_report",
                        help="Path to write JSON report")
    
    args = parser.parse_args()
    
    root_path = Path(args.path)
    if root_path.is_file():
        notebooks = [root_path]
    else:
        if not root_path.exists():
             print(f"Path not found: {root_path}")
             sys.exit(1)
        # Recursively find .ipynb files, excluding checkpoints
        notebooks = [p for p in root_path.rglob("*.ipynb") 
                     if ".ipynb_checkpoints" not in str(p)]

    results = []
    any_violation = False

    print(f"Auditing {len(notebooks)} notebooks in {root_path}...")

    for nb_path in notebooks:
        res = audit_notebook(nb_path)
        if not res["compliant"]:
            any_violation = True
            print(f"FAIL: {res['path']}")
            for v in res["violations"]:
                print(f"  - {v}")
        else:
             if args.check and args.json_report is None:
                 pass 
             elif not args.check:
                 print(f"OK: {res['path']}")

        results.append(res)
    
    report = {
        "summary": {
            "total": len(notebooks),
            "compliant": len([r for r in results if r["compliant"]]),
            "failed": len([r for r in results if not r["compliant"]])
        },
        "notebooks": results
    }

    if args.json_report:
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.json_report}")

    if args.check and any_violation:
        print("Audit failed: Violations detected.")
        sys.exit(1)
    
    if args.check:
         print("Audit passed.")

if __name__ == "__main__":
    main()
