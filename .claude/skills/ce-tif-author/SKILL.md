---
name: ce-tif-author
description: >
  Create or revise CE TIF specifications and scenarios with structured
  public-API observations.
---

# CE TIF Author

## Use this skill when

- Authoring a new `CE-TIF-<AREA>-<NNN>.md` TIF specification
- Implementing or revising `tif_<area>.py` scenario code
- Adding observation fields to cover new requirement acceptance criteria
- Updating the TIF README index table with a new TIF row

## Inputs

- Requirement ID(s) to be served (CE-REQ-... files from `development/capabilities/requirements/`)
- Governing ADR(s)
- Public API surface under test (WrapCalibratedExplainer method(s))
- Applicable CE task types and fixture requirements

## Workflow

### Part A — TIF specification (CE-TIF-AREA-NNN.md)

1. **Determine the TIF ID.** Use `CE-TIF-<AREA>-<NNN>` matching the requirement area.
   Check the TIF README index in `development/capabilities/verification/tif/README.md`
   for the next available number.

2. **Fill the Identity table:** `tif_id`, `executable`, `entry_functions`,
   `evidence_builder`, `evidence_key`, `verification_type`, `status`.

3. **List requirements served** with a column for which observation fields each
   requirement uses (the link from observation → acceptance criterion).

4. **List claims served** (CE-CAP-... for navigation; not required for correctness).

5. **Define the fixture / data contract.** Must be:
   - Deterministic (seeded random state)
   - No external data, network, or clock access
   - Consistent with `sklearn.datasets.make_classification` or `make_regression`
     unless a real-world fixture is required and documented

6. **Write the WrapCalibratedExplainer workflow.** Show the exact call sequence:
   `fit → calibrate → [predict/predict_proba/explain_factual/explore_alternatives/...]`

7. **Define observation fields.** Each field must:
   - Capture exactly one observable property
   - Have a name that mirrors the requirement acceptance criterion it closes
   - Be documented with type and what it captures

8. **Define acceptance fields.** Map requirement IDs to the observation fields
   and the assertion that must hold (e.g. `exception_raised must be False`).

9. **Define evidence fields.** List the fields that must appear in raw evidence
   records (minimum: commit_sha, package_version, test_id, dataset_id, random_seed, result).

### Part B — Python scenario (tif_area.py)

10. **Create or update the Python scenario.** The scenario must:
    - Import `WrapCalibratedExplainer` from the public CE API
    - Use a dataclass or typed dictionary as the observation return type
    - Implement `run_<area>_tif_scenario()` as the entry function
    - Implement `build_evidence_payload(obs)` that returns a dict with all evidence fields
    - Not perform final pytest assertions (local sanity checks only)
    - Not import from `calibrated_explanations.core.calibrated_explainer` directly
    - Not construct explanation objects directly (no `FactualExplanation(...)`)
    - Not use any private members (`._something`)

### Part C — Update indexes

11. **Update the TIF README index table** in `tif/README.md` with a new row:
    `| TIF ID | spec link | executable link | evidence key | verification type | status | requirements | claims |`

## Verification

```bash
python scripts/quality/validate_capability_chain.py
make capability-chain-check
```

## Output contract

Return:

1. Complete `CE-TIF-<AREA>-<NNN>.md` specification file.
2. The `tif_<area>.py` Python scenario (new or updated).
3. Updated TIF README index row.
4. A summary: TIF ID, requirements served, observation fields added.

## Constraints

- **Non-negotiable**: every TIF scenario must enter through `WrapCalibratedExplainer`.
  Never bypass fit→calibrate lifecycle.
- TIF scenarios must not construct explanation objects directly.
- TIF scenarios must not use private members or internal CE imports.
- Observation fields must be observable (externally visible) — no internal state.
- Each requirement must map to at least one observation field in the spec.
- `verification_type` must be one of: `api_contract`, `behavioral_contract`,
  `numerical_behavior`, `empirical_smoke`.
- The TIF is not a pytest test — it must not call `pytest.assert` or `assert`
  on final results. Local sanity pre-conditions are permitted.
- Fixture data must be deterministic: no `random_state=None`, no network calls,
  no file system reads outside the repo.
