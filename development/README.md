> **Active scope:** Canonical development documentation map; the single entry point for maintainer planning, ADRs, Standards, and governance navigation. Remains active indefinitely; updated when the layout changes or new canonical locations are added.

# Development Documentation Map

This map defines the internal development documentation surface for
Calibrated Explanations.

`docs/` is for user-facing and contributor-facing documentation. `development/`
is for maintainer planning, engineering governance, capability claim and
requirement catalogs, and curated closure evidence summaries.

---

## Canonical Locations

| Material | Location |
|---|---|
| Active development planning | `development/current-work/` |
| Forward-looking planning | `development/future-work/` |
| Closed planning and curated closure evidence summaries | `development/finished-work/` |
| Architectural Decision Records | `development/adrs/` |
| Engineering Standards | `development/standards/` |
| ADR-030 quality method tooling | `development/standards/test-quality-method/` |
| CI/tooling JSON schemas (non-runtime) | `development/schemas/` |
| Capability claims | `development/capabilities/claims/` |
| Requirements derived from capability claims | `development/capabilities/requirements/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability verification | `tests/capabilities/` |
| Generated verification run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |

These locations are authoritative even when a directory has not yet been
created. Do not create additional locations for the same material.

---

## Verification Layout Rule

Capability verification follows this chain:

```text
Capability claim
    -> requirement
    -> verification case
    -> evidence record
```

Use the location map above for each layer:

- Claims belong in `development/capabilities/claims/`.
- Requirements belong in `development/capabilities/requirements/`.
- Executable verification scenarios and helpers belong in
  `development/capabilities/verification/`.
- Pytest checks belong in `tests/capabilities/` for new capability-contract
  tests. Existing nearby unit or integration tests may be linked from
  requirements when they already verify the required public behavior.
- Generated run outputs belong in `reports/verification/`.
- Human-curated release or closure evidence summaries belong in
  `development/capabilities/evidence/`.

Do not add claim catalogs, requirement catalogs, schemas, verification code,
tests, or evidence records unless a task explicitly asks for that work.

---

## Legacy Notice

`docs/improvement/` and `docs/standards/` were legacy entry points. Both have
been fully removed (Task 8, v0.11.4). All active planning, ADRs, Standards,
test-quality-method documents, CI/tooling schemas, and PlotSpec evidence records
now live exclusively under `development/`. Do not recreate these directories.

---

## Navigation

Start here, then follow the current locations:

1. Active release plan: `development/current-work/RELEASE_PLAN_v1.md`
2. ADRs: `development/adrs/`
3. Standards: `development/standards/`
4. Test quality method (ADR-030 tooling): `development/standards/test-quality-method/`
5. Test guidance: `tests/README.md`
6. Repository-wide agent rules: `CONTRIBUTOR_INSTRUCTIONS.md`
