# Companion Repository Design

Target: `kristinebergs/calibrated-explanations-plugins`, the official public companion
repository named in ADR-010's 2026-07-24 addendum and in `CONTRIBUTING.md`.

The previous revision of this design proposed a single flat
`src/calibrated_explanations_plugins/` package. That does not match the repository,
which is a monorepo of independently versioned distributions. This revision is written
against the actual layout, verified 2026-09-20.

## Actual repository conventions

- Packages live at `packages/<family>/calibrated-explanations-<family>-<name>/`.
- Families are fixed: `scripts/validate_repo_structure.py` hard-codes
  `PLUGIN_FAMILIES = ("calibration", "explanation", "visualization")`, plus `meta`.
- Each package builds with hatchling, has its own version, and declares
  `[tool.ce_plugin_repo]` with `family`, `status`, and `import_name`.
- Lifecycle is metadata, not location: `experimental`, `mature`, `deprecated`
  (ADR-P001, the plugins repo's own ADR at `docs/adr/ADR-P001-plugin-lifecycle-and-curation.md`
  in that repo, numbered separately from CE ADRs). Curation is expressed only through
  family metapackage dependencies.
- Entry points use the `calibrated_explanations.plugins` group, with family-specific
  groups such as `...plugins.plot_builders` where applicable.
- Existing metadata uses `"provider": "official"` and sets both `trusted: False` and
  `trust: False`.

There is no narrative or LLM code in the repository today, so nothing is duplicated.

## Open question: which family

A narrative provider fits none of the three existing families. Two options, and the
maintainers must choose before any package is scaffolded:

1. **Add a `narrative` family.** Requires updating `PLUGIN_FAMILIES`, the umbrella
   metapackage, and amending ADR-P001's "exactly three family metapackages" statement.
   Cleanest conceptually. ADR-P001 also cites CE's superseded ADR-014 as a governing
   runtime contract; the amendment should correct that to ADR-037.
2. **Place it under `explanation`.** No structure change, but narration is not an
   explanation method, and ADR-041 D10 is explicit that narration performs no analysis.
   This would blur the distinction the CE-side ADR is built on.

Option 1 is the better fit. It is a repository governance decision, not a CE one.

## Proposed package

Assuming option 1:

```text
packages/narrative/calibrated-explanations-narrative-llm/
├── pyproject.toml
├── README.md
├── MATURITY.md
├── src/ce_narrative_llm/
│   ├── __init__.py
│   ├── plugin.py            # NarrativeProvider implementation, entry-point target
│   ├── clients.py           # injected client protocol and adapters
│   ├── prompts.py           # versioned prompts
│   ├── structured_output.py # parse to CandidateNarrative with placeholders
│   ├── redaction.py
│   ├── repair.py
│   ├── policy.py
│   └── errors.py
└── tests/
```

```toml
[project]
name = "calibrated-explanations-narrative-llm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["calibrated-explanations>=<first release with ADR-041>"]

[project.optional-dependencies]
openai = ["<provider-sdk>"]

[project.entry-points."calibrated_explanations.plugins"]
narrative_llm = "ce_narrative_llm.plugin:StructuredLLMNarrativeProvider"

[tool.ce_plugin_repo]
family = "narrative"
status = "experimental"
import_name = "ce_narrative_llm"
```

Base install imports no LLM SDK. The client is an injected protocol so the provider is
testable without a vendor SDK or network.

## Provider behaviour

1. Receive a validated `NarrativeSpec` and `NarrativeRequest`.
2. Apply the redaction the request permits.
3. Build a versioned structured prompt.
4. Call an injected client under explicit timeout and attempt limits.
5. Parse structured output into `CandidateNarrative`, emitting
   `{{fact:<id>.<field>}}` placeholders for every numeric, sign, and interval claim.
6. Optionally repair using CE verifier findings.
7. Return the candidate for mandatory host verification.

It must not accept raw explanation objects or PlotSpec, infer interval semantics from
numbers, access the estimator or calibration data, aggregate local explanations, call
itself verified or calibrated, trust itself on import, or silently switch model.

## Provider controls

- Secrets are runtime-injected and absent from `repr`, logs, caches, specs, and results.
- Retention and training settings are documented and default conservatively.
- Timeouts, attempts, and token or cost limits are bounded.
- Raw vendor responses are debug-only, off by default, and never in stable results.
- Repair receives only the minimised spec and bounded CE verifier findings.
- Network tests are opt-in; standard CI uses a fake client and makes zero live calls.

## Test matrix

| Layer | Required tests | Gate |
|---|---|---|
| Metadata | supported and unsupported CE plugin and spec versions | fail before generation on mismatch |
| Packaging | base and optional extras, lazy imports | no SDK import before provider use |
| Structured output | valid and invalid claims, fact references, placeholders | deterministic parse or typed failure |
| Behaviour | timeout, cancellation, attempts, missing extra, fake client | bounded deterministic policy |
| Repair | correctable and uncorrectable host findings | never overrides a host failure |
| Security | injection, markup, Unicode, secrets, oversize | no execution, leak, or bypass |
| Provenance | model, prompt, settings, network, attempts | complete and reproducible |

## Research assets belong elsewhere

The previous revision placed `research/fixtures`, `research/metrics`, and
`research/experiments` in this repository. Under the ADR-010 addendum, research
evaluation and reproduction assets live in
`kristinebergs/calibrated-explanations-studies`. Locked `NarrativeSpec` fixtures,
faithfulness metrics, and human-study exports go there.

The plugin package keeps only its own contract tests and a fake client.

## Publication path

Per `CONTRIBUTING.md` and the plugins repository README:

1. Open a plugin-intake request on the Moffran repository.
2. Accepted work enters this repository as **experimental**, source-only, not published
   and not curated.
3. Promotion to `mature` uses the maturity-promotion PR template, automated gates, and
   maintainer review.
4. Publication is governed by tag validation and the protected PyPI environment.

Submitting a request does not authorise publication. A networked provider would be the
first of its kind here; `SECURITY.md` currently says nothing about network access,
credentials, or secrets, and should be extended before such a package is published.

## Versioning

The package versions its own implementation, its prompt and structured-output contract,
its supported CE plugin API range, and its supported `NarrativeSpec` range. It does not
version or reinterpret CE explanation payloads, because the host performs that mapping.

## Initial release boundary

v0.1 is one structured provider, an injected fake client, redaction, timeout and repair
policy, provenance, and adversarial tests. It does not duplicate the CE template
baseline and ships no global aggregation.
