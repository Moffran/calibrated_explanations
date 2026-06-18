.PHONY: test test-cov

# Run the full unit test suite with the default coverage configuration.
test:
	pytest -q

# Local test target (no coverage) kept for quick runs.
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

# Run only the new CI entrypoints (keeps legacy duplicates out of the run).
.PHONY: ci-local-new
ci-local-new:
	python scripts/run_ci_locally.py --shell bash --workflow ci-pr --workflow ci-full --workflow ci-main --workflow ci-nightly

.PHONY: check-private-members
check-private-members:
	python scripts/anti-pattern-analysis/scan_private_usage.py --check

.PHONY: check-agent-instructions
check-agent-instructions:
	python scripts/quality/check_agent_instruction_consistency.py

.PHONY: check-report-paths
check-report-paths:
	python scripts/quality/check_no_local_paths_in_reports.py --check --report reports/quality/no_local_paths_report.json

.PHONY: check-ci-policy
check-ci-policy:
	# Includes full-SHA action pin enforcement, reusable-workflow checks, and release-docs gate.
	# Reusable workflows (reusable-*.yml) are exempt from heavy-gating; they are gated by callers.
	python scripts/quality/validate_ci_policy.py --base-sha HEAD~1 --head-sha HEAD --advisory

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
# local_checks_pr will remain "unavailable" — only CI can set it after running the full suite.
.PHONY: governance-status-local
governance-status-local:
	python scripts/quality/build_governance_status_artifact.py --output reports/governance/governance_status.json --validate --run-lint

# Run stacked CI-equivalent checks in the current Python environment,
# including `pre-commit run --all-files` (no install/bootstrap steps).
.PHONY: local-checks local-checks-pr
local-checks:
	python scripts/local_checks.py

.PHONY: local-checks-ci
local-checks-ci:
	python scripts/local_checks.py --ci-parity

# PR-scope only: lint/type/core-tests + policy scanners.
local-checks-pr:
	python scripts/local_checks.py --skip-main
