.PHONY: test test-cov

# Run the full unit test suite with the default coverage configuration.
test:
	pytest -q --cov=src/calibrated_explanations --cov-config=pyproject.toml --cov-report=term-missing --cov-report=xml --cov-fail-under=90

# Explicit local coverage gate.
test-cov:
	pytest -q --cov=src --cov-report=xml:coverage.xml --cov-fail-under=90

# Fast test target for core modules (excludes viz-marked tests).
.PHONY: test-core
test-core:
	@home="$${HOME:-$${USERPROFILE:-$${HOMEDRIVE}$${HOMEPATH}}}"; \
	if [ -z "$$home" ]; then \
		if pwd -W >/dev/null 2>&1; then home="$$(pwd -W)"; else home="$$PWD"; fi; \
	fi; \
	export HOME="$$home"; \
	export USERPROFILE="$$home"; \
	if echo "$$home" | grep -q '^[A-Za-z]:'; then \
		export HOMEDRIVE="$${home%%:*}:"; \
		export HOMEPATH="$${home#*:}"; \
	fi; \
	export MPLCONFIGDIR="$$home/.matplotlib"; \
	export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \
	pytest -o addopts= -m "not viz"

# Run only viz-marked tests (useful when the `[viz]` extras are installed).
.PHONY: test-viz
test-viz:
	pytest -m viz --no-cov

# Run a local CI dry-run that lists CI steps discovered by the helper script.
.PHONY: ci-local-dry-run
ci-local-dry-run:
	python scripts/run_ci_locally.py --dry-run

# Run a full local CI run that executes all steps discovered by the helper script.
.PHONY: ci-local
ci-local:
	python scripts/run_ci_locally.py

# Run the workflow run-block smoke helper (executes local `run:` blocks only).
.PHONY: ci-local-runblocks
ci-local-runblocks:
	python scripts/local_checks.py --ci-parity

.PHONY: check-private-members
check-private-members:
	python scripts/anti-pattern-analysis/scan_private_usage.py --check

.PHONY: check-agent-instructions
check-agent-instructions:
	python scripts/quality/check_agent_instruction_consistency.py

.PHONY: check-report-paths
check-report-paths:
	python scripts/quality/check_no_local_paths_in_reports.py --check --report reports/quality/no_local_paths_report.json

# Blocking full-inventory CI-policy validation (ADR-035, v0.11.6 Task 60).
# The identical command runs in the ci.yml policy job, the local-checks-pr
# profile, and the .github/-scoped pre-commit hook.
.PHONY: check-ci-policy
check-ci-policy:
	python scripts/quality/validate_ci_policy.py --full-inventory

.PHONY: uv-install-smoke
uv-install-smoke:
	python scripts/local_checks.py --uv-install-smoke

.PHONY: adr030-ratification
adr030-ratification:
	python scripts/local_checks.py --adr030-ratification

.PHONY: warning-policy
warning-policy:
	python scripts/quality/check_warning_policy.py --check --report reports/quality/warning_policy.json

.PHONY: check-extras-parity
check-extras-parity:
	python scripts/quality/check_core_extras_parity.py --check --report reports/quality/core_extras_parity.json

.PHONY: deprecation-closure
deprecation-closure:
	python scripts/local_checks.py --deprecation-closure

.PHONY: pydocstyle
pydocstyle:
	python -m pydocstyle src tests

# CI mode: lint flags must be supplied by the caller (e.g. from workflow step exit codes).
.PHONY: governance-status
governance-status:
	python scripts/quality/build_governance_status_artifact.py --output reports/governance/governance_status.json --validate

# Local mode: runs ruff and mypy, captures their exit codes, then writes the artifact.
# local_checks_pr will remain "unavailable" - only CI can set it after running the full suite.
.PHONY: governance-status-local
governance-status-local:
	python scripts/quality/build_governance_status_artifact.py --output reports/governance/governance_status.json --validate --run-lint

# Run local verification profiles in the current Python environment.
.PHONY: quick local-checks local-checks-quick local-checks-task local-checks-pr local-checks-full local-checks-release local-checks-ci
quick:
	python scripts/local_checks.py --profile quick

local-checks-quick:
	python scripts/local_checks.py --profile quick

local-checks-task:
	python scripts/local_checks.py --profile task --task $(TASK)

local-checks:
	# Compatibility alias: this is the heavy local gate, not a routine inner-loop command.
	python scripts/local_checks.py --profile full

local-checks-full:
	python scripts/local_checks.py --profile full

local-checks-ci:
	# Compatibility alias for workflow run-block smoke; not full GitHub Actions parity.
	python scripts/local_checks.py --ci-parity

# PR-scope only: quick checks + blocking PR gates.
local-checks-pr:
	python scripts/local_checks.py --profile pr

local-checks-release:
	python scripts/local_checks.py --profile release

# Prepare deterministic release files while version-dependent task checklists
# are still open. The strict preflight repeats this step and remains mandatory.
.PHONY: release-prepare-files
release-prepare-files:
	python scripts/local_checks.py --release-prepare-files $(if $(VERSION),--release-version $(VERSION),) $(if $(RELEASE_DATE),--release-date $(RELEASE_DATE),)

# Pre-tag release gate: release.md steps 1-10, including deterministic release
# file updates, strict validation, build, artifact checks, and wheel smoke.
# VERSION=X.Y.Z and RELEASE_DATE=YYYY-MM-DD are optional inference overrides.
.PHONY: release-preflight
release-preflight:
	python scripts/local_checks.py --release-preflight $(if $(VERSION),--release-version $(VERSION),) $(if $(RELEASE_DATE),--release-date $(RELEASE_DATE),)

# Guard the manual publish phase against stale or incomplete preflight state.
.PHONY: release-finalize
release-finalize:
	python scripts/local_checks.py --release-finalize

# Post-publish steps (release.md 14-17): PyPI metadata check, clean-env install
# smoke, release-plan handoff/archive, and development-version bump.
# Run after steps 11-13 (commit/tag/push, RTD publish, PyPI upload) are done manually.
# NEXT_VERSION=<milestone> is optional; the master plan is authoritative by default.
.PHONY: release-postcommit
release-postcommit:
	python scripts/local_checks.py --release-postcommit $(if $(NEXT_VERSION),--next-version $(NEXT_VERSION),) $(if $(RELEASE_DATE),--release-date $(RELEASE_DATE),)
	python scripts/update_conda_feedstock.py

# Optional, best-effort: bump the conda-forge feedstock recipe to the latest
# PyPI release and open a PR. Runs automatically at the end of
# release-postcommit; exposed standalone to retry after a failure. No-op
# unless CE_CONDA_FEEDSTOCK_DIR points at a local feedstock clone.
.PHONY: conda-feedstock-update
conda-feedstock-update:
	python scripts/update_conda_feedstock.py

# Validate the capability verification chain without executing TIF scenarios.
# Safe to run on every PR - does not mutate any files.
.PHONY: capability-chain-check
capability-chain-check:
	python scripts/quality/validate_capability_chain.py --check
	python scripts/generate_tif_evidence.py --validate-existing

# Regenerate raw evidence by running all TIF scenarios and checking they pass at HEAD.
# Run explicitly when TIF behavior or acceptance logic changes, and at release closure.
.PHONY: capability-evidence-refresh
capability-evidence-refresh:
	python scripts/generate_tif_evidence.py --check-current
