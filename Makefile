.PHONY: test test-cov

# Run the full unit test suite with the default coverage configuration.
test:
	pytest

# Mirror the CI coverage invocation (coverage options are centralised in pytest.ini).
test-cov:
	pytest

# Run a local CI dry-run that lists CI steps discovered by the helper script.
.PHONY: ci-local-dry-run
ci-local-dry-run:
	python scripts/run_ci_locally.py --dry-run

# Run a full local CI run that executes all steps discovered by the helper script.
.PHONY: ci-local
ci-local:
	python scripts/run_ci_locally.py
