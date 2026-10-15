# TIF - Test Interface Framework

> **Governing ADR:** `development/adrs/ADR-040-capability-verification-framework.md`
> (see D5–D6 for the TIF layer and exemption rules). On conflict, ADR-040 is
> authoritative; this README is the operational guide.

This directory contains TIF (Test Interface Framework) verification interface
specifications and executable TIF scenarios for CE capability verification.

TIF is the layer between requirements and tests. It is not pytest. TIF defines
the interface between requirements and executable verification.

See `development/capabilities/verification/README.md` for the full TIF definition.

## What TIF is

A TIF interface:

1. Stimulates CE exclusively through `WrapCalibratedExplainer`.
2. Captures structured observations from the stimulus.
3. Returns those observations as a typed dataclass or dictionary.
4. Does not perform final pytest assertions (except local sanity checks).
5. Enables multiple tests and requirements to share one verified stimulus.

## What TIF is not

- TIF is not a pytest test. Tests call TIF; TIF observes.
- TIF is not a fixture. TIF is parameterized by test scenarios, not by pytest fixture injection.
- TIF is not a mock. TIF runs the actual CE public workflow.

## Non-negotiable TIF rule

Every behavioral CE TIF scenario must stimulate CE through `WrapCalibratedExplainer`.

TIF scenarios must not:
- Construct explanation objects directly
- Use private/internal CE APIs (`._private`, `CalibratedExplainer` imports)
- Bypass the `fit -> calibrate -> explain` lifecycle

## File naming

```text
CE-TIF-<AREA>-<NNN>.md    # interface specification
tif_<area>.py              # executable scenario implementing the spec
```

## Current TIF Interfaces

| TIF ID | Specification | Executable | Evidence Key | Verification Type | Status | Requirements served | Claims served |
|---|---|---|---|---|---|---|---|
| CE-TIF-EXPL-001 | [CE-TIF-EXPL-001.md](CE-TIF-EXPL-001.md) | [tif_explanation.py](tif_explanation.py) | EXPL-001 | behavioral_contract | active | CE-REQ-EXPL-API-001, CE-REQ-EXPL-RETURN-001, CE-REQ-EXPL-API-002, CE-REQ-EXPL-ALT-RETURN-001 | CE-CAP-EXPL-001, CE-CAP-EXPL-002 |
| CE-TIF-EXPL-CONJ-001 | [CE-TIF-EXPL-CONJ-001.md](CE-TIF-EXPL-CONJ-001.md) | [tif_conjunction.py](tif_conjunction.py) | EXPL-CONJ-001 | behavioral_contract | active | CE-REQ-EXPL-CONJ-API-001, CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-RULE-001, CE-REQ-EXPL-CONJ-PARAM-001 | CE-CAP-EXPL-CONJ-001 |
| CE-TIF-FILTER-001 | [CE-TIF-FILTER-001.md](CE-TIF-FILTER-001.md) | [tif_filter.py](tif_filter.py) | FILTER-001 | api_contract | active | CE-REQ-EXPL-FILTER-SUPER-001, CE-REQ-EXPL-FILTER-SEMI-001, CE-REQ-EXPL-FILTER-COUNTER-001, CE-REQ-EXPL-FILTER-ENSURED-001, CE-REQ-EXPL-FILTER-PARETO-001 | CE-CAP-EXPL-FILTER-001 |
| CE-TIF-GUARD-001 | [CE-TIF-GUARD-001.md](CE-TIF-GUARD-001.md) | [tif_guard.py](tif_guard.py) | GUARD-001 | api_contract | active | CE-REQ-GUARD-API-001 | CE-CAP-GUARD-001 |
| CE-TIF-MOND-001 | [CE-TIF-MOND-001.md](CE-TIF-MOND-001.md) | [tif_mondrian.py](tif_mondrian.py) | MOND-001 | api_contract | active | CE-REQ-MOND-API-001, CE-REQ-MOND-CONS-001, CE-REQ-MOND-VAL-001, CE-REQ-MOND-LIFE-001, CE-REQ-MOND-SER-001 | CE-CAP-MOND-001 |
| CE-TIF-NARR-001 | [CE-TIF-NARR-001.md](CE-TIF-NARR-001.md) | [tif_narrative.py](tif_narrative.py) | NARR-001 | api_contract | active | CE-REQ-NARR-API-001 | CE-CAP-NARR-001 |
| CE-TIF-PRED-001 | [CE-TIF-PRED-001.md](CE-TIF-PRED-001.md) | [tif_prediction.py](tif_prediction.py) | PRED-001 | behavioral_contract | active | CE-REQ-PRED-API-001, CE-REQ-PRED-INTERVAL-BOUNDS-001 | CE-CAP-PRED-001 |
| CE-TIF-PRED-CLASS-001 | [CE-TIF-PRED-CLASS-001.md](CE-TIF-PRED-CLASS-001.md) | [tif_classification.py](tif_classification.py) | PRED-CLASS-001 | numerical_behavior | active | CE-REQ-PRED-CLASS-API-001, CE-REQ-PRED-CLASS-BOUNDS-001 | CE-CAP-PRED-CLASS-001 |
| CE-TIF-PRED-PROB-001 | [CE-TIF-PRED-PROB-001.md](CE-TIF-PRED-PROB-001.md) | [tif_prob_regression.py](tif_prob_regression.py) | PRED-PROB-001 | numerical_behavior | active | CE-REQ-PRED-PROB-API-001, CE-REQ-PRED-PROB-BOUNDS-001 | CE-CAP-PRED-PROB-001 |
| CE-TIF-PRED-PROB-002 | [CE-TIF-PRED-PROB-002.md](CE-TIF-PRED-PROB-002.md) | [tif_prob_regression_instance.py](tif_prob_regression_instance.py) | PRED-PROB-002 | numerical_behavior | active | CE-REQ-PRED-PROB-INSTANCE-001 | CE-CAP-PRED-PROB-001 |
| CE-TIF-REJECT-001 | [CE-TIF-REJECT-001.md](CE-TIF-REJECT-001.md) | [tif_reject.py](tif_reject.py) | REJECT-001 | api_contract | active | CE-REQ-REJECT-API-001 | CE-CAP-REJECT-001 |
| CE-TIF-VIZ-001 | [CE-TIF-VIZ-001.md](CE-TIF-VIZ-001.md) | [tif_visualization.py](tif_visualization.py) | VIZ-001 | empirical_smoke | active | CE-REQ-VIZ-SMOKE-001 | CE-CAP-VIZ-001 |

## Chain enforcement

Run after any change to claims, requirements, TIF specs, tests, or evidence:

```bash
make capability-chain-check
```

This runs the non-mutating chain validator and validates committed evidence files.

Run at release closure or when TIF behavior changes:

```bash
make capability-evidence-refresh
```

This re-executes all TIF scenarios, writes fresh evidence, and fails if evidence is not at HEAD.

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated raw run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
| Capability-chain validator | `scripts/quality/validate_capability_chain.py` |
| Raw evidence generator | `scripts/generate_tif_evidence.py` |
