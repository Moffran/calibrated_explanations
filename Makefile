.PHONY: test test-cov

# Run the full unit test suite with the default coverage configuration.
test:
pytest

# Mirror the CI coverage invocation so contributors can reproduce gate failures locally.
test-cov:
pytest --cov=src/calibrated_explanations --cov-report=term-missing --cov-fail-under=90
