# Run a fast subset of tests by enabling FAST_TESTS and lowering SAMPLE_LIMIT
$env:FAST_TESTS = "1"
$env:SAMPLE_LIMIT = "200"
pytest
