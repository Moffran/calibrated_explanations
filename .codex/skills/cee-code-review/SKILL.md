---
name: cee-code-review
description: >
  Review CEE code changes for package isolation, V2 protocol compliance, OSS terminology, CE-First invariants, and enterprise coding conventions.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Code Review — Core Instructions

# CEE Code Review

## Use this skill when
- About to submit a PR and need a final review
- Reviewing someone else's PR on the CEE repo
- After writing new enterprise code (adaptive, governance, common)
- The CI is green but you want a standards check
- Asked "does this code follow CEE standards?"

## Inputs
- The diff or files to review
- `AGENTS.md` — CEE coding conventions and critical rules
- `development/002_engineering_standards.md` — authoritative engineering standards

## Review Dimensions

Run all dimensions. Flag BLOCKING issues that must be fixed before merge.

### Dimension 1: Package Isolation (BLOCKING)
```bash
grep -r "from calibrated_explanations_enterprise.adaptive" packages/common/
grep -r "from calibrated_explanations_enterprise.governance" packages/common/
grep -r "from calibrated_explanations_enterprise.adaptive" packages/governance/
grep -r "from calibrated_explanations_enterprise.governance" packages/adaptive/
```
All must return empty. → Use `cee-package-isolation` for remediation.

### Dimension 2: CE-First Compliance (BLOCKING)
- No mocks or stubs for `calibrated_explanations` imports
- No `MagicMock()` replacing `WrapCalibratedExplainer` in non-unit tests
- Hard `ImportError` raised if library is missing (not silent fallback)
- Parity tests use `np.allclose(rtol=0, atol=1e-10)` for offline CE calls

### Dimension 3: OSS Terminology (BLOCKING)
- No "counterfactual" in new code (use "alternatives")
- No "Orchestrator" (use "Manager")
- V2 protocol uses `"features"`, `"predictions"`, `"uncertainty_low"`, `"uncertainty_high"`
- Internal code uses sklearn-style `X`, `y`, `X_cal`, `y_cal`

### Dimension 4: Future Annotations (REQUIRED)
```python
from __future__ import annotations  # Must be first import in every source file
```

### Dimension 5: Lazy Imports (REQUIRED)
- `matplotlib`, `pandas`, `joblib` must NOT appear at module top-level in files reachable from `__init__.py`
- Import inside functions or use `TYPE_CHECKING` guard

### Dimension 6: Exception Handling (REQUIRED)
- No bare `Exception` or `ValueError` without documented reason
- Use `ValidationError` or `CheckpointError` from `common/exceptions.py`
- No silent exception swallowing

### Dimension 7: Frozen Configs (REQUIRED)
- All Pydantic config models must have `model_config = ConfigDict(frozen=True)`
- Config objects must be immutable after creation

### Dimension 8: Clean Entry Points (REQUIRED)
- `main.py` and server entry points must NOT contain business logic
- All orchestration delegated to manager classes
- Entry points: imports + app instantiation + manager delegation only

### Dimension 9: No Runtime Artifacts (REQUIRED)
```bash
git status  # must show no .parquet, .pkl, .db files staged
```

### Dimension 10: Google-style Docstrings (ADVISORY)
- Public methods in enterprise classes should have Google-style docstrings
- Args, Returns, Raises documented

### Dimension 11: Type Hints (REQUIRED)
- All function signatures have type hints
- `from typing import Any, Dict, List, Literal, Optional, Protocol` for legacy compat
- No untyped `def` in public API

### Dimension 12: Logging (REQUIRED)
- Use `get_enterprise_logger()` from governance telemetry
- No bare `print()` in production code
- Structured log context for enterprise events

## Verification
```bash
ruff check .
pytest -q
```

## Output contract
Return a structured review report:
```
## CEE Code Review Report

### BLOCKING Issues (must fix before merge)
- [ ] <issue description> at <file:line>

### REQUIRED Issues (should fix)
- [ ] <issue description> at <file:line>

### ADVISORY Notes
- <note>

### Summary
PASS / FAIL (FAIL if any BLOCKING issues exist)
```

## Constraints
- Use the OSS `ce-code-review` skill for OSS-layer concerns (ADRs, CE internals)
- This skill focuses on CEE-specific rules that `ce-code-review` doesn't cover
- A review is complete only when all BLOCKING issues are resolved
- If parity tests are affected, also invoke `cee-parity-test`