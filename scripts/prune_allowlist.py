import runpy
from pathlib import Path

if __name__ == "__main__":  # pragma: no cover - shim
    runpy.run_path(str(Path(__file__).parent / "quality" / "prune_allowlist.py"), run_name="__main__")
    raise SystemExit(0)

import json

path = Path(r"c:\Users\loftuw\Documents\Github\kristinebergs-calibrated_explanations\.github\private_member_allowlist.json")
data = json.loads(path.read_text(encoding="utf-8"))
original_count = len(data["allowlist"])

# Filter for legacy only
new_allowlist = [
    entry for entry in data["allowlist"]
    if entry["file"].replace("\\", "/").startswith("tests/legacy/")
]

data["allowlist"] = new_allowlist
path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

print(f"Pruned allowlist from {original_count} to {len(new_allowlist)} entries.")
