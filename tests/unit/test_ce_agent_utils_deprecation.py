"""ADR-011 deprecation contract for ``calibrated_explanations.ce_agent_utils``.

Each scenario runs in a fresh interpreter so module import caching and the
global warning registry of the pytest process cannot mask or leak the warning.
Delete this file together with the module in v1.3.0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = "calibrated_explanations.ce_agent_utils"
EXPECTED_MESSAGE = (
    "calibrated_explanations.ce_agent_utils is deprecated since v1.1.0 and will be "
    "removed in v1.3.0. Use WrapCalibratedExplainer and the documented public "
    "Calibrated Explanations API directly."
)

_RECORD_IMPORT = f"""
import json, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import {MODULE}
print(json.dumps([[w.category.__name__, str(w.message)] for w in caught]))
"""

_NORMAL_WORKFLOW = f"""
import json, sys, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from calibrated_explanations import WrapCalibratedExplainer

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    wrapper = WrapCalibratedExplainer(LogisticRegression())
    wrapper.fit(x[:60], y[:60])
    wrapper.calibrate(x[60:110], y[60:110])
    wrapper.predict_proba(x[110:], uq_interval=True)
    wrapper.explain_factual(x[110:112]).to_narrative(output_format="text")
print(json.dumps({{
    "loaded": "{MODULE}" in sys.modules,
    "messages": [str(w.message) for w in caught if "ce_agent_utils" in str(w.message)],
}}))
"""


def _run(code: str, *, strict: bool = False) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if k not in {"CE_DEPRECATIONS", "PYTEST_CURRENT_TEST"}
    }
    if strict:
        env["CE_DEPRECATIONS"] = "error"
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_should_emit_deprecation_warning_when_module_is_imported() -> None:
    # Act
    result = _run(_RECORD_IMPORT)

    # Assert
    assert result.returncode == 0, result.stderr
    caught = json.loads(result.stdout.strip().splitlines()[-1])
    assert ["DeprecationWarning", EXPECTED_MESSAGE] in caught


def test_should_raise_when_module_is_imported_under_strict_deprecations() -> None:
    # Act
    result = _run(f"import {MODULE}", strict=True)

    # Assert
    assert result.returncode != 0
    assert f"DeprecationWarning: {EXPECTED_MESSAGE}" in result.stderr


def test_should_not_load_or_warn_when_normal_public_workflow_runs() -> None:
    # Act
    result = _run(_NORMAL_WORKFLOW)

    # Assert
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"loaded": False, "messages": []}
