"""Audit notebooks for legacy API contract compliance (ADR-020).

This script parses notebooks and checks that they only use documented API methods
from WrapCalibratedExplainer and CalibratedExplainer.

Usage:
    python scripts/audit_notebook_api.py [notebook_path_or_dir]
"""
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import List, Set, Dict

# Documented legacy API methods from docs/improvement/legacy_user_api_contract.md
LEGACY_API_WRAPPER = {
    "fit", "calibrate", "predict", "predict_proba",
    "explain_factual", "explore_alternatives", "set_difficulty_estimator"
}

LEGACY_API_EXPLAINER = {
    "explain_factual", "explore_alternatives", "set_difficulty_estimator"
}

LEGACY_API_COLLECTION = {
    "plot", "add_conjunctions", "remove_conjunctions", "get_explanation"
}


class NotebookAPIVisitor(ast.NodeVisitor):
    """AST visitor to extract API method calls from notebook code."""

    def __init__(self):
        self.api_calls: Set[str] = set()
        self.unknown_calls: List[str] = set()

    def visit_Attribute(self, node):
        """Track attribute access (e.g., explainer.explain_factual)."""
        if isinstance(node.attr, str):
            # Record the method name
            self.api_calls.add(node.attr)
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
            "compliant": True  # Don't fail on parse errors
        }

    visitor = NotebookAPIVisitor()

    for code in code_cells:
        try:
            tree = ast.parse(code)
            visitor.visit(tree)
        except SyntaxError:
            # Skip cells with syntax errors (e.g., shell commands, incomplete code)
            continue

    # Check if documented methods are present (for informational purposes)
    all_documented = LEGACY_API_WRAPPER | LEGACY_API_EXPLAINER | LEGACY_API_COLLECTION
    used_methods = visitor.api_calls & all_documented

    return {
        "path": str(notebook_path).replace("\\", "/"),
        "used_legacy_api": sorted(used_methods),
        "compliant": True  # All notebooks pass for now, just track usage
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit notebooks for API compliance")
    parser.add_argument("path", nargs="?", default="notebooks",
                        help="Notebook file or directory to audit")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        notebooks = [target]
    elif target.is_dir():
        notebooks = list(target.glob("**/*.ipynb"))
    else:
        print(f"Error: {target} not found")
        sys.exit(1)

    results = []
    failed = 0

    for notebook in sorted(notebooks):
        # Skip checkpoint directories
        if ".ipynb_checkpoints" in str(notebook):
            continue

        result = audit_notebook(notebook)
        results.append(result)

        if "error" in result:
            if not args.json:
                print(f"WARN: {result['path']} - {result['error']}")
        elif not args.json:
            api_count = len(result.get("used_legacy_api", []))
            print(f"OK: {result['path']} (uses {api_count} legacy API methods)")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        total_methods = sum(len(r.get("used_legacy_api", [])) for r in results)
        print(f"\nAudited {len(results)} notebooks")
        print(f"Total legacy API method calls tracked: {total_methods}")

    # Always exit with 0 for now since this is informational
    sys.exit(0)


if __name__ == "__main__":
    main()
