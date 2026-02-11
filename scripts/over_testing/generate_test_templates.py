"""Generate minimal test templates for identified coverage gaps.

The script creates `tests/generated/test_cov_fill_<n>.py` files with a small
template referencing the file and line range to be covered. These are templates
for manual completion (they intentionally do NOT implement black-box assertions
automatically). Use them as a starting point when adding minimal focused tests.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


TEMPLATE = '''# Auto-generated template to address coverage gap in {file}
# Lines: {start}-{end} (length={length})
import importlib
import types
import pytest

def test_fill_gap_{idx}():
    """Placeholder test for {file} lines {start}-{end}.

    This template attempts a safe package-style import of the target
    module and skips the test if the module is not importable in this
    environment.
    """
    module_name = "{module}"
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        pytest.skip("module not importable: " + module_name)
    assert isinstance(mod, types.ModuleType)

'''


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gaps-csv", required=True, help="CSV of gaps as produced by gap_analyzer: file,start,end,len")
    p.add_argument("--out-dir", default="tests/generated", help="output directory for templates")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    with open(args.gaps_csv, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            file, start, end, length = parts[0], parts[1], parts[2], parts[3]
            idx += 1
            # derive an importable module name
            fp = file.replace("\\", "/")
            module = None
            if "calibrated_explanations/" in fp:
                module = fp.split("calibrated_explanations/", 1)[1]
                module = module.rsplit(".py", 1)[0].lstrip("./").replace("/", ".")
                module = f"calibrated_explanations.{module}"
            elif fp.startswith("src/"):
                tail = fp.split("src/", 1)[1]
                if tail.startswith("calibrated_explanations/"):
                    module = tail.split("calibrated_explanations/", 1)[1]
                    module = module.rsplit(".py", 1)[0].lstrip("./").replace("/", ".")
                    module = f"calibrated_explanations.{module}"
            elif "/" in fp:
                module = fp.rsplit(".py", 1)[0].lstrip("./").replace("/", ".")
                module = f"calibrated_explanations.{module}"
            else:
                module = f"calibrated_explanations.{fp.rsplit('.py',1)[0]}"

            fname = out_dir / f"test_cov_fill_{idx:03d}.py"
            with open(fname, "w", encoding="utf-8") as of:
                of.write(TEMPLATE.format(file=file, start=start, end=end, length=length, idx=idx, module=module))
            print(f"wrote {fname}")


if __name__ == "__main__":
    main()
