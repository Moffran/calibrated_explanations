.PHONY: test test-cov

# Run the full unit test suite with the default coverage configuration.
test:
	pytest

# Mirror the CI coverage invocation (coverage options are centralised in pytest.ini).
test-cov:
	pytest
