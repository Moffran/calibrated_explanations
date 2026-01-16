.PHONY: test test-cov

# Run the full unit test suite with the default coverage configuration.
test:
	pytest

# Mirror the CI coverage invocation (coverage options are centralised in pytest.ini).
test-cov:
	pytest

# Fast test target for core modules (excludes viz-marked tests).
.PHONY: test-core
test-core:
	export HOME="$USERPROFILE" && pytest -m "not viz" --cov=src --cov-report xml

# Run only viz-marked tests (useful when the `[viz]` extras are installed).
.PHONY: test-viz
test-viz:
	pytest -m viz

# Run a local CI dry-run that lists CI steps discovered by the helper script.
.PHONY: ci-local-dry-run
ci-local-dry-run:
	python scripts/run_ci_locally.py --dry-run

# Run a full local CI run that executes all steps discovered by the helper script.
.PHONY: ci-local
ci-local:
	python scripts/run_ci_locally.py

.PHONY: check-private-members
check-private-members:
	python scripts/anti-pattern-analysis/scan_private_usage.py --check
