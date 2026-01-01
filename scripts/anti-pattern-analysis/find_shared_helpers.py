import ast
import os
import collections
from pathlib import Path

def find_shared_private_helpers():
    defs = collections.defaultdict(list)
    for p in Path('tests').rglob('*.py'):
        try:
            tree = ast.parse(p.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('_') and not node.name.startswith('__'):
                    defs[node.name].append(str(p))
        except Exception:
            continue

    print("Shared private test helpers (defined in multiple files):")
    for name, files in sorted(defs.items()):
        if len(set(files)) > 1:
            print(f"{name}: {len(set(files))} files - {list(set(files))[:3]}...")

if __name__ == "__main__":
    find_shared_private_helpers()
