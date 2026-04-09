# Notebook Policy for `calibrated_explanations`

> **Scope:** Rules for notebooks under `notebooks/` that are executed in CI
> via `scripts/docs/run_notebooks.py`. Cross-links to the release checklist
> and ADR-012 rather than redefining their ownership.
>
> **Governing ADR:** `docs/improvement/adrs/ADR-012-documentation-and-gallery-build-policy.md`

---

## 1. Contribution rules

Every notebook contributed to `notebooks/` MUST:

1. **Run headlessly** — no interactive widgets, no `%matplotlib inline` that
   requires a display.  Use `import matplotlib; matplotlib.use("Agg")` or rely
   on the project's headless CI configuration.
2. **Seed all randomness** — call `numpy.random.seed(0)` (or equivalent) near
   the top of the notebook so outputs are reproducible across runs.
3. **Use light datasets** — data loading must complete in ≤ 5 s and the
   full notebook must finish within the CI ceiling (default: 300 s per
   notebook, 30 s per cell).
4. **Be importable with `[notebooks,viz]` extras only** — notebooks MUST NOT
   depend on packages outside the `[notebooks,viz]` extra group.  Core
   dependencies MUST NOT be expanded (ADR-010).

---

## 2. Skip tags (`metadata.ce_skip`)

Set the `ce_skip` field in the notebook-level `metadata` block to opt out of
execution:

```json
{
  "metadata": {
    "ce_skip": "noexec"
  }
}
```

| Tag value | Effect | When to use |
|-----------|--------|-------------|
| `"noexec"` | Always skipped; emits `skipped_noexec` in report. | Notebooks that demonstrate interactive features, require GPU, or are permanently excluded from CI. |
| `"slow"` | Skipped; emits `skipped_slow` in report. Review required before release. | Notebooks estimated > 5 min (e.g. training loops, large datasets). |

**Policy violation:** Any unknown `ce_skip` value is a policy violation.  In
blocking mode (release branches) the notebook is recorded as `failed`.  In
advisory mode a `UserWarning` is raised and execution proceeds.

---

## 3. Runtime ceilings (ADR-012)

| Parameter | Default | Override via CLI |
|-----------|---------|-----------------|
| Per-cell timeout | 30 s | `--cell-timeout N` |
| Per-notebook wall-clock | 300 s | `--notebook-timeout N` |

When a ceiling is exceeded the notebook record receives `status: "timed_out"`.
Raising either ceiling requires a PR rationale comment because it affects all
notebooks in scope.

---

## 4. Execution modes (ADR-012 branch-gate contract)

| Context | Mode | CLI flag | Enforcement |
|---------|------|----------|-------------|
| mainline (`main`) | `advisory` | `--mode advisory` | Nightly CI reports failures but never blocks. |
| Before release | `blocking` | `--mode blocking` | Maintainer runs blocking mode before tagging; any `failed` or `timed_out` notebook must be resolved first. |

The nightly CI job (`notebook-exec-report` in `ci-nightly.yml`) always runs in
advisory mode. Release gating is the responsibility of the release manager.

Before cutting a release, run the driver locally
in blocking mode to catch failures early:

```bash
python scripts/docs/run_notebooks.py \
  --mode blocking \
  --output reports/docs/notebook_execution_report.json
```

Resolve all `failed` and `timed_out` entries before tagging the release.

---

## 5. Report schema

The driver writes `reports/docs/notebook_execution_report.json`.  Each record
carries:

| Field | Type | Description |
|-------|------|-------------|
| `notebook` | `str` | Relative path to the notebook. |
| `status` | `str` | One of: `passed`, `failed`, `timed_out`, `skipped_noexec`, `skipped_slow`. |
| `elapsed_seconds` | `float` | Wall-clock seconds. |
| `errors` | `list` | List of `{cell, etype, evalue}` dicts; empty on success. |
| `invocation_id` | `str` | UUID shared across all records in one run. |
| `skip_reason` | `str \| null` | Tag value used to skip, or `null`. |
| `mode` | `str` | `"advisory"` or `"blocking"`. |

Validate any custom tooling against the schema with:

```python
from scripts.docs.run_notebooks import validate_report_schema
errs = validate_report_schema(report_dict)
assert errs == []
```

---

## 6. Cross-links

- **ADR-012** — governing policy: `docs/improvement/adrs/ADR-012-documentation-and-gallery-build-policy.md`
- **ADR-010** — optional-dependency boundary: `docs/improvement/adrs/ADR-010-optional-dependency-split.md`
- **Release checklist** — resolve notebook failures before branching: `docs/improvement/RELEASE_PLAN_v1.md`
- **CI nightly job** — `notebook-exec-report` in `.github/workflows/ci-nightly.yml`
