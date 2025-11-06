# Documentation test policy

Every Python example published in the documentation must execute unchanged as
part of the automated test suite. The maintainer workflow mirrors the directory
structure under `docs/` so each page with a `python` code block has a matching
module under `tests/docs/`.

## Principles

1. **Copy code verbatim.** Tests for documentation examples should copy the code
   blocks directly from the source page. Avoid rewriting the snippet so the
   exercises remain faithful to what readers run locally.
2. **Mirror the docs hierarchy.** Place tests in `tests/docs/<path-to-page>` so
   the file tree matches the documentation layout. This keeps navigation obvious
   when auditing coverage.
3. **Cover the README.** The README quickstart follows the same rule—create a
   `tests/test_readme_doc.py` module unless a quickstart test already exercises
   the identical snippet.
4. **Gracefully skip optional extras.** When examples rely on optional
   dependencies (`prometheus_client`, PlotSpec, pandas, etc.), guard the test
   with `pytest.importorskip` so missing extras do not break baseline CI runs.

## Maintainer checklist

- Add or update the corresponding test whenever a documentation snippet
  changes.
- Update `docs/foundations/governance/test_policy` when the strategy for
  organising documentation tests evolves.
- Confirm `pytest tests/docs` runs cleanly before promoting changes that touch
  docs or README examples.
