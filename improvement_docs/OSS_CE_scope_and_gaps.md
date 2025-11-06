> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v0.9.0 maintenance review

# OSS scope inventory and gap tracking

Document decisions that scope optional surfaces for the OSS distribution and
capture outstanding improvements.

## Streaming-friendly explanation delivery (v0.9.0)

- **Status:** Deferred for v0.9.0. Batch exports remain the supported path while
  we validate the new JSON helpers.
- **Owner:** Runtime tech lead (release gate signer).
- **Memory profile:** Baseline probabilistic regression exports (10k rows) stay
  under 200 MB when using `CalibratedExplanations.to_json()` in batches of 256.
  Larger exports should be chunked manually until the generator prototype
  graduates.
- **Follow-up milestone:** Target revisit in v0.9.1 after collecting telemetry
  from early adopters.
- **Notes:** The deferral keeps calibration semantics stable while the team
  prototypes streamed JSON writers and optional chunked CLI helpers. Guidance is
  documented in `docs/how-to/export_explanations.md` so users know to batch their
  exports and capture metadata alongside the payload.
