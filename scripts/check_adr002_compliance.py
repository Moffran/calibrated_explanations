import ast
import sys
from pathlib import Path

DISALLOWED_EXCEPTIONS = {
    "ValueError",
    "RuntimeError",
    "Exception",
    "TypeError",
}

DISALLOWED_WARNINGS = {
    "RuntimeWarning",
}

def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return []

    violations = []
    for node in ast.walk(tree):
        # Check for disallowed exceptions in Raise
        if isinstance(node, ast.Raise):
            exc_name = None
            if node.exc is None:
                continue # bare raise is okay (re-raise)

            if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                exc_name = node.exc.func.id
            elif isinstance(node.exc, ast.Name):
                exc_name = node.exc.id

            if exc_name in DISALLOWED_EXCEPTIONS:
                violations.append((node.lineno, f"Raised disallowed exception: {exc_name}. Use CalibratedError or subclasses per ADR-002."))

        # Check for disallowed exceptions in ExceptHandler
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                continue

            names_to_check = []
            if isinstance(node.type, ast.Name):
                names_to_check.append(node.type.id)
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        names_to_check.append(elt.id)

            for name in names_to_check:
                if name in DISALLOWED_EXCEPTIONS:
                    violations.append((node.lineno, f"Caught disallowed exception: {name}. This may mask root causes; prefer specific handling or CalibratedError types if applicable."))

        # Check for warnings.warn usage
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'warn':
                # Check if it's warnings.warn (heuristic)

                # Check keywords for stacklevel
                has_stacklevel = False
                for keyword in node.keywords:
                    if keyword.arg == 'stacklevel':
                        has_stacklevel = True
                        break

                if not has_stacklevel:
                     violations.append((node.lineno, "warnings.warn() called without 'stacklevel' argument (ADR-002 best practice)."))

                # Check category
                category = None
                if len(node.args) > 1:
                    if isinstance(node.args[1], ast.Name):
                        category = node.args[1].id

                # Check keywords for category
                for keyword in node.keywords:
                    if keyword.arg == 'category':
                        if isinstance(keyword.value, ast.Name):
                            category = keyword.value.id


                if category in DISALLOWED_WARNINGS:
                    violations.append((node.lineno, f"Used disallowed warning category: {category}. Use specific warnings or CalibratedError types."))

    return violations


def main():
    src_dir = Path("src/calibrated_explanations")
    all_violations = {}

    print(f"Scanning {src_dir.resolve()}...")
    count = 0
    for filepath in src_dir.rglob("*.py"):
        count += 1
        violations = check_file(filepath)
        if violations:
            all_violations[str(filepath)] = violations

    print(f"Scanned {count} files.")

    if all_violations:
        print("ADR-002 Violations Found:")
        for filepath, violations in all_violations.items():
            print(f"\nFile: {filepath}")
            for line, msg in violations:
                print(f"  Line {line}: {msg}")
        sys.exit(1)
    else:
        print("No ADR-002 violations found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
