# Remedy List for Generated Tests

This file lists generated `test_cov_fill_*` tests that require manual remediation
to conform to ADR-030 and repository test-quality rules.

Summary (auto-generated):

- Total `test_cov_fill_*` scanned: see `reports/over_testing/cov_fill_adr30_scan.csv`.
- Prune plan produced at `reports/over_testing/prune_plan.json` (conservative: no automatic removals proposed).

All generated files are currently flagged as *questionable* by the conservative pruning heuristic because they contain assertions and therefore may be meaningful; human review is required to decide whether each is:

- **Keep & Move:** the test is behavior-first and conforms to ADR-030 — move to `tests/auto_approved/` and rename accordingly.
- **Refactor:** the test is useful but tests private internals or is non-deterministic — refactor to test public behavior per ADR-030.
- **Remove:** the test is a trivial placeholder or duplicates other tests — move to `reports/over_testing/backup_removed_tests/` or delete after confirmation.

Next steps (manual):

1. Open `reports/over_testing/cov_fill_adr30_scan.csv` and inspect rows marked `has_assertion=False` first (none currently).
2. For each file listed under `prune_plan.json` → `questionable`, review test contents and decide action (Keep/Refactor/Remove).
3. Record per-file decisions in this document (append) and run `python scripts/over_testing/prune_generated_tests.py --apply` to apply deletions once reviewed.

This remedy list must be reviewed and signed off by a core maintainer before any mass removals.
