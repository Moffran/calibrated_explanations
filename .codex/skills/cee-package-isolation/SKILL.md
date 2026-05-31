---
name: cee-package-isolation
description: >
  Verify and enforce CEE's common/adaptive/governance dependency isolation rules before and after any cross-package change.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Package Isolation — Core Instructions

# CEE Package Isolation

## Use this skill when
- Adding a new import to any CEE package
- Moving code between `common`, `adaptive`, or `governance`
- CI is failing with an import error related to cross-package imports
- Reviewing a PR that touches package `__init__.py` or adds dependencies
- Asked "can package X import from package Y?"

## Inputs
- `AGENTS.md` §"Package Dependency Policy" and §"Architecture Patterns"
- The file(s) being added or modified
- `packages/*/pyproject.toml` — each package's declared dependencies

## Dependency Rules

```
common   ──► nothing (no adaptive, no governance)
adaptive ──► common only (no governance)
governance ──► common only (no adaptive)
```

These rules guarantee:
- Independent deployability (governance-only install doesn't pull adaptive deps)
- No accidental feature coupling
- CI validates isolation by running tests per-package

## Workflow

1. **Check what package the file lives in** — determine from its path:
   - `packages/common/src/...` → common package
   - `packages/adaptive/src/...` → adaptive package
   - `packages/governance/src/...` → governance package
   - `src/calibrated_explanations_enterprise/...` → top-level entry point (can import all)

2. **Run isolation checks** (must return no results):

```bash
grep -r "from calibrated_explanations_enterprise.adaptive" packages/common/
grep -r "from calibrated_explanations_enterprise.governance" packages/common/
grep -r "from calibrated_explanations_enterprise.adaptive" packages/governance/
grep -r "from calibrated_explanations_enterprise.governance" packages/adaptive/
```

3. **If a violation is found**:
   - Identify the import and why it was added
   - Determine whether the shared type/utility belongs in `common` instead
   - Move the shared code to `common` if needed
   - Update both packages to import from `common`

4. **Check pyproject.toml** for each affected package — the `install_requires` or `dependencies` list must not include a sibling package (adaptive or governance must not list each other as deps)

5. **Verify namespace extension** — each package `__init__.py` must have:
   ```python
   __path__ = __import__('pkgutil').extend_path(__path__, __name__)
   ```

## Verification
```bash
# All four must return no results:
grep -r "from calibrated_explanations_enterprise.adaptive" packages/common/
grep -r "from calibrated_explanations_enterprise.governance" packages/common/
grep -r "from calibrated_explanations_enterprise.adaptive" packages/governance/
grep -r "from calibrated_explanations_enterprise.governance" packages/adaptive/

# Run tests after any package change:
pytest -q
```

## Output contract
Return:
1. Isolation check results (pass/fail for each of the four grep checks)
2. List of any violations found with file:line references
3. Remediation steps for each violation (move to common, or restructure)
4. Confirmation that `pyproject.toml` deps are correct

## Constraints
- Never add a direct import from adaptive in governance or vice versa
- Shared types, protocols, and utilities MUST go in common
- If you find yourself needing adaptive in governance, reconsider the design
- The top-level `src/calibrated_explanations_enterprise/` entry point is the only place that can compose across all packages
