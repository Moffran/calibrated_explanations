> **Active scope:** Governing architectural decision for feature input validation, categorical encoding, and the calibration-set mapping contract at `CalibratedExplainer` entry points. Remains active as long as this contract governs CE data ingestion; superseded when the policy is revised.

> **Status note (2026-09-04):** Last edited 2026-09-04 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-009: Input Preprocessing & Mapping Persistence Policy

Status: Accepted
Date: 2025-08-22
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Issue reported: The library only supports numeric input natively. Users must call
`utils.helper.transform_to_numeric` manually to encode DataFrames with text/categorical
features and manage mappings externally. Native support in the wrapper would improve
usability and reproducibility.

## Decision

- Keep the core numeric; add preprocessing in the wrapper layer:
  - `wrap_explainer.py` learns/applies preprocessing (either built-in `transform_to_numeric` or a user-supplied transformer/pipeline).
  - Add configuration: `auto_encode=True|False|'auto'`, `preprocessor: Optional[Transformer]`, and policy for unseen categories (`'ignore'|'error'`).
  - `auto_encode='auto'` is the automatic encoding mode; it deterministically
    learns mappings during fit/init and applies them during inference.
  - Default unseen-category policy is `'error'`; `'ignore'` is an explicit opt-in.
  - Persist mapping artifacts on the explainer; attach mapping metadata to Explanation provenance.
  - Provide mapping persistence helpers:
    `Explainer.export_mapping() -> dict` and `Explainer.import_mapping(mapping: dict) -> None`.
    Mapping primitives must be JSON-safe for serialization.
- Validation (`core/validation.py`) detects DataFrames/non-numeric columns and enforces NaN/dtype policies, returning actionable errors.

## Consequences

- Positive: greatly improved ergonomics; deterministic mappings in predict/online; clearer provenance.
- Negative: additional complexity in wrapper; need to document behavior and storage.


## Governed claims

- `CE-CAP-PREPROC-001` — Wrapper preprocessing learns and reuses deterministic feature mappings while keeping the core numeric and preserving provenance boundaries.

## Alternatives

- Enforce user-supplied preprocessing only (status quo), which is less friendly and harder to reproduce.
- Push preprocessing into core classes, which couples responsibilities and complicates testing/design.

## Adoption & Migration

- Phase 1B: extend validation to be DataFrame-aware.
- Phase 2: introduce wrapper preprocessing options, persist mappings, and round-trip tests; document configs.

### Adoption Progress (2025-09-02)

Implemented:

- User-supplied preprocessor wiring with private fit/transform helpers in `wrap_explainer.py`.
- Config fields in `ExplainerConfig` (`preprocessor`, `auto_encode`) and builder pass-through.
- Docs updated (Getting Started, API reference) to show config-driven preprocessing.
- Tests added for deterministic reuse of the same transformer across fit/inference.

Pending:

- Automatic encoding path (`auto_encode='auto'`) and mapping persistence helpers
  for that mode with deterministic mapping storage on the wrapper.
- Unseen-category policy behavior and documentation (default `'error'`,
  configurable `'ignore'`).
- Mapping export/import helpers with JSON-safe primitives and documentation
  examples.

## Open Questions

- Where to store mappings (in-memory only vs. optional serialization helpers)?
  Start in-memory with API hooks for export/import.

## Implementation status (2026-06-02, v0.11.3 Task 9 Workstream B)

**Gap 2 — JSON-safe mapping export (closed):** `WrapCalibratedExplainer.export_preprocessor_mapping()` (in `src/calibrated_explanations/core/wrap_explainer.py`) already enforces JSON-safe output via `_validate_json_safe_mapping()` on both the `get_mapping_snapshot` and `mapping_` fallback paths. `import_preprocessor_mapping()` also validates JSON safety on import. Tests in `tests/unit/core/test_wrap_explainer_helpers.py` verify enforcement. The public helper names are `export_preprocessor_mapping` / `import_preprocessor_mapping` (not `export_mapping` / `import_mapping` as originally proposed — placement on wrapper is deliberate per ADR-001 boundary rules).

**Gap 3 — Helper-placement doc drift (closed):** The ADR-009 §Decision describes `Explainer.export_mapping()` and `Explainer.import_mapping()`. The actual implementation exposes these as `WrapCalibratedExplainer.export_preprocessor_mapping()` and `WrapCalibratedExplainer.import_preprocessor_mapping()`. This naming differs from the ADR proposal text but is intentional: the wrapper is the public preprocessing API surface (ADR-001 boundary), and the more descriptive names distinguish mapping persistence from other wrapper export operations. No code change is required; this note records the deliberate divergence so future contributors do not chase a naming drift as a bug.

**Post-v1.0 open item — `transform_to_numeric` root-namespace export and deprecation (deferred):**
`transform_to_numeric` is currently exported from the root `calibrated_explanations` namespace (`__all__`). It predates the wrapper preprocessing API (this ADR) and was the only option for users who needed to encode categorical DataFrames before passing data to `CalibratedExplainer`. With `WrapCalibratedExplainer` now providing `auto_encode`, `preprocessor`, and `export_preprocessor_mapping()`/`import_preprocessor_mapping()`, the standalone utility function is largely redundant for the main user-facing workflow.

However, until the `auto_encode='auto'` path and full mapping persistence UX (pending items above) are complete, `transform_to_numeric` remains a necessary fallback for users who call `CalibratedExplainer` directly (not via the wrapper). Post-v1.0 scope:
- Complete the `auto_encode='auto'` mapping persistence path so the wrapper is the sole recommended preprocessing entry point.
- Deprecate `transform_to_numeric` from the root namespace (move to `calibrated_explanations.utils` for users who need it explicitly).
- Remove from `__all__` in a v1.1+ ADR-011 deprecation cycle.

## Adoption Progress (2026-09-04, v1.0.1 T4 / issue #202)

ADR consult for T4 (v1.0.1 plan) surfaced two gaps this ADR left unaddressed
for the `auto_encode='auto'` built-in encoder path: missing-value handling,
and numeric-vs-categorical column detection for non-DataFrame input. Both are
resolved below so implementation and requirements-authoring (`CE-REQ-PREPROC-*`)
have a governing decision to build against.

**Decision — configuration home: Session tier (`ExplainerConfig`), not
`ConfigManager`.** All configuration introduced by this addendum —
`missing_value_policy` and the `categorical_features` override described
below — is added to `ExplainerConfig`/`ExplainerBuilder`
(`calibrated_explanations.api.config`), the same object that already holds
`preprocessor`, `auto_encode`, and `unseen_category_policy` per this ADR's
2025-09-02 adoption note. This is a Session-tier concern under ADR-038's
four-tier taxonomy (deployment / session / strategy / tuning): it is
per-dataset, per-explainer behavioral configuration fixed once at
`WrapCalibratedExplainer`/`CalibratedExplainer` construction, not deployment
configuration resolved from environment variables or `pyproject.toml`.

`ConfigManager` (ADR-034) was considered and rejected for this role.
ADR-034 §"Capabilities governed by ConfigManager" scopes it to deployment
configuration — plugin selection, telemetry mode, cache/parallel settings,
feature-filter settings, strict observability, CI markers — resolved through
an env-var → `pyproject.toml` → call-site precedence chain and snapshotted
once per process/CLI invocation. `categorical_features` (column indices of a
specific `X`) and `missing_value_policy` vary per dataset and per explainer
instance, not per deployment; there is no meaningful environment-variable or
`pyproject.toml` default for "which columns of this dataset are
categorical." Routing them through `ConfigManager` would also violate
ADR-038's rule that the four tiers "cannot override" one another by
collapsing a Session-tier concept into the Deployment tier. `ExplainerConfig`
is the ADR-038-compliant, already-established home for this exact class of
setting.

**Decision — missing-value policy.** A new, independent `missing_value_policy`
field is added to `ExplainerConfig`/`ExplainerBuilder`, alongside the
existing `unseen_category_policy` field (default `'error'`, opt-in
`'ignore'`). It is a distinct axis, not a value of `unseen_category_policy`,
because "value never seen during fit" and "value absent from this row" are
different conditions with different safe defaults.

- `'category'` — a missing value (`NaN`/`None`) in a categorical column is
  mapped to a deterministic sentinel category (e.g. `"__missing__"`), learned
  and persisted like any other observed category. The sentinel must be
  canonicalized/escaped so it cannot collide with a real observed category
  value.
- `'error'` — raise `ValidationError` on missing values, matching the
  existing numeric NaN-rejection default used elsewhere in the codebase
  (`core/validation.py`, `allow_nan=False`).

Proposed default: **`'category'`**, on the grounds that missing categorical
values are common in realistic mixed-type tabular data and `auto_encode='auto'`
exists specifically to provide a zero-setup convenience path — an `'error'`
default would silently defeat that purpose for a large share of real
datasets. **This default is a proposal, not yet maintainer-confirmed; treat
it as open until confirmed in requirements-authoring or by explicit
maintainer sign-off.**

**Decision — numeric-vs-categorical detection on non-DataFrame input.** For
pandas `DataFrame` input, dtype already distinguishes numeric from
object/categorical/string/boolean columns unambiguously (existing scope, no
change). For raw array input (e.g. a plain `numpy` array, as in issue #202's
own reproduction case), detection is defined as:

1. **Default auto-detection: per-column castability.** Attempt numeric
   coercion per column; a column that fully coerces is treated as numeric and
   passed through untouched, a column that does not is encoded categorically.
   This is a dtype/castability test, not a semantic or cardinality-based
   guess, and therefore stays inside issue #202's explicit non-goal
   ("Automatically guessing semantic categories from arbitrary numeric
   values"). A column of numeric-looking strings that denote coded categories
   (e.g. ZIP-like codes) will be classified as numeric passthrough under this
   rule; this is a disclosed, accepted limitation of castability-based
   detection, not a defect to remediate later.
2. **Explicit override: reuse the existing `categorical_features` parameter,
   promoted to `ExplainerConfig`.** `CalibratedExplainer.__init__` already
   accepts `categorical_features` as a list of column indices; that name and
   index-based semantics are reused as-is — no new parameter is introduced.
   What changes is *where it is first captured*: `ExplainerConfig`/
   `ExplainerBuilder` gains a `categorical_features` field so the value is
   fixed at `WrapCalibratedExplainer`/`CalibratedExplainer` construction
   (`from_config`) rather than being known only once `calibrate(**kwargs)`
   runs. `calibrate(**kwargs)` MAY still accept a per-call
   `categorical_features` override for `CalibratedExplainer.__init__`'s
   existing discretization/perturbation use (unrelated to the encoder), but
   the built-in encoder's detection reads exclusively from the
   session-level `ExplainerConfig` value, which is always available before
   both `fit()` and `calibrate()`. When supplied, listed indices are always
   treated as categorical by the built-in encoder regardless of castability;
   unlisted indices fall back to rule 1.

**Resolution of the fit()/calibrate() timing gap.** Promoting
`categorical_features` to `ExplainerConfig` (this addendum) resolves the
sequencing question raised when this decision was first drafted:
`ExplainerConfig` is fixed at construction, strictly before both `fit()`
(which needs the hint to encode `x_proper_train` before the learner is
fit) and `calibrate()` (which needs the same hint to encode
`x_calibration`). Both encoding passes read the same, already-known,
immutable Session-tier value — no reconciliation rule between two arrival
times is needed because there is only one arrival time.

**Persistence note.** Both decisions change what `BuiltinEncoder.get_mapping_snapshot()`
stores (per-column numeric-passthrough/categorical classification, and a
missing-value sentinel entry where applicable). Per ADR-031, this is an
incompatible change to the `preprocessing_mapping.json` artifact shape and
requires a `schema_version` increment there, with the same fail-fast
rejection of old-schema artifacts already used for the wrapper's own state
schema. ADR-031's versioning rules govern this; they are not restated here.
