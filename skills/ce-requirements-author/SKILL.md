---
name: ce-requirements-author
description: >
  Create or revise testable CE requirements with acceptance criteria,
  verification targets, and evidence fields.
---

# CE Requirements Author

## Use this skill when

- Writing a new `CE-REQ-*.md` requirement file derived from a capability claim
- Revising acceptance criteria, scope, or verification targets in an existing requirement
- Adding a TIF reference to an existing requirement after the TIF is authored
- Updating verification_status after a test is created and verified

## Inputs

- Parent claim ID(s) (CE-CAP-... references)
- Governing ADR(s) and/or STD(s)
- Public API surface under obligation (WrapCalibratedExplainer method signatures)
- Applicable CE task types (binary_classification, regression, etc.)

## Workflow

1. **Determine the requirement ID.** Use the convention `CE-REQ-<AREA>-<FACET>-<NNN>`.
   Check existing files in `development/capabilities/requirements/` for the next number.
   AREA and FACET match the parent claim's area (e.g. EXPL → CE-REQ-EXPL-API-001).

2. **Define the obligation scope.** Identify:
   - Which public API method(s) are under obligation
   - Applicable CE task types
   - Valid workflow states (fitted + calibrated? pre-fit error cases only?)

3. **Write observable behavior.** Enumerate what the system must do when the obligation
   applies. Use numbered, observable statements:
   - "Return without raising an exception for valid inputs"
   - "Return an object that supports indexing to retrieve per-instance explanations"
   Each statement should map 1-to-1 to an acceptance criterion.

4. **Write the acceptance criterion.** State the pass/fail condition as a concrete
   assertion using CE public API terms. Example:
   - `len(explain_factual(X_test)) == len(X_test)`
   - `explain_factual(X_test)[0]` is not `None`

5. **Assign the verification method.** Choose from: automated pytest test,
   analysis, inspection. Provide the `test_id` and file path if a test exists.

6. **List evidence required.** Minimum: `commit_sha`, `package_version`, `test_id`,
   `dataset_id`, `random_seed`, `result`.

7. **Write the assumption boundary.** State what the requirement does NOT verify
   (e.g. statistical validity, coverage guarantees, correctness of magnitudes).

8. **Add `tif_refs` field.** Leave as `[]` if no TIF has been authored yet;
   fill in the CE-TIF-... ID once the TIF is created.

## Verification

```bash
python scripts/quality/validate_capability_chain.py --requirements-only
```

## Output contract

Return:

1. The complete `CE-REQ-<AREA>-<FACET>-<NNN>.md` file contents with all sections.
2. A one-line summary: requirement ID, obligation type, parent claim ID, verification method.
3. Any companion claim file update needed (adding this req ID to `requirements:` list).

## Constraints

- `obligation_type` must be one of: `api_contract`, `behavioral_contract`,
  `numerical_behavior`, `governance_constraint`.
- `claim_refs` must list the parent `CE-CAP-...` ID — upward traceability is mandatory.
- `tif_refs` must list `CE-TIF-...` IDs, not test IDs — TIFs are the bridge layer.
- `verification_status` must be `unverified` on creation; only set to `verified` once
  a passing test exists and evidence has been recorded.
- Acceptance criteria must be stated in terms of CE public API behavior, not internal
  implementation details.
- Do not write requirements that accept only the happy path — boundary and error
  conditions must each have their own requirement or be explicitly excluded in scope.
