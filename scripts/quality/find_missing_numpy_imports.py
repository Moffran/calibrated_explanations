import os
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "src"
missing = []
for dirpath, dirs, files in os.walk(root):
    for f in files:
        if not f.endswith('.py'):
            continue
        path = Path(dirpath) / f
        with path.open('r', encoding='utf-8') as fh:
            txt = fh.read()
        if re.search(r'\bnp\.', txt) or re.search(r'\bnp\b', txt):
            if 'import numpy as np' not in txt:
                missing.append(path)

print('Files using np but missing import:')
for p in missing:
    print(p.as_posix())
