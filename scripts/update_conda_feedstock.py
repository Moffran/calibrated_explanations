"""Optional post-release trigger for the conda-forge feedstock version bump.

Runs the feedstock's own `update_meta.py` (fetch latest PyPI release, update
recipe/meta.yaml, commit, push, open a PR against conda-forge/main) against a
local clone of https://github.com/tuvelofstrom/calibrated-explanations-feedstock.

This step is best-effort and never fails `make release-postcommit`: it only
runs when CE_CONDA_FEEDSTOCK_DIR points at an existing local clone, and a
failure here is reported as a warning, not an error, since it touches a
separate external repository outside this project's release contract
(release.md steps 14-17).

Configure once per machine (not committed to this repo):
    setx CE_CONDA_FEEDSTOCK_DIR "C:\\path\\to\\calibrated-explanations-feedstock"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    feedstock_dir = os.environ.get("CE_CONDA_FEEDSTOCK_DIR")
    if not feedstock_dir:
        print(
            "[conda feedstock] CE_CONDA_FEEDSTOCK_DIR is not set; skipping the "
            "feedstock version bump. Set it to a local clone of "
            "tuvelofstrom/calibrated-explanations-feedstock to enable this step."
        )
        return 0

    repo_path = Path(feedstock_dir)
    update_script = repo_path / "update_meta.py"
    if not update_script.is_file():
        print(
            f"[conda feedstock] CE_CONDA_FEEDSTOCK_DIR={feedstock_dir!r} does not "
            "contain update_meta.py; skipping the feedstock version bump."
        )
        return 0

    print(f"[conda feedstock] Running {update_script} ...")
    result = subprocess.run([sys.executable, str(update_script)], cwd=repo_path, text=True)
    if result.returncode != 0:
        print(
            "[conda feedstock] WARNING: the feedstock version bump failed "
            f"(exit {result.returncode}). This does not block the release; rerun "
            f"manually with:\n  cd {feedstock_dir}\n  python update_meta.py"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
