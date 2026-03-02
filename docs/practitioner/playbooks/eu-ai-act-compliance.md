# Satisfying EU AI Act Requirements with `calibrated_explanations`

> [!WARNING]
> **LEGAL DISCLAIMER — NO COMPLIANCE GUARANTEE**
>
> This document and this library do not constitute legal advice and carry no
> guarantee that any regulatory authority or legal entity will regard the
> library's outputs as satisfying any statutory obligation. Whether a deployment
> satisfies the EU AI Act (Regulation (EU) 2024/1689) or any other law is a
> **legal determination requiring qualified legal counsel**. The library is
> provided "as is" under the BSD 3-Clause licence. **The authors and maintainers
> expressly disclaim all liability arising from reliance on this document.**

> **Document scope:** This guide targets practitioners and compliance officers who
> deploy machine learning models in contexts regulated by Regulation (EU) 2024/1689
> (the EU AI Act). It maps each relevant article to the concrete capabilities of
> `calibrated_explanations` and provides a verifiable implementation checklist.

---

## 1. Executive Summary

The EU AI Act (Regulation (EU) 2024/1689) imposes legally enforceable obligations
on providers and deployers of high-risk AI systems, including requirements for
transparency, uncertainty documentation, human oversight, and the right of affected
persons to receive meaningful explanations. Accuracy metrics alone — precision,
recall, AUC — do not satisfy these obligations. A model that achieves 95 % accuracy
on a held-out test set still cannot, by itself, tell a regulator *why* a specific
individual was denied credit, *how confident* the system was at the time of that
decision, or *what change* to the individual's circumstances would have produced a
different outcome. The AI Act requires precisely this per-instance, traceable,
uncertainty-aware information.

`calibrated_explanations` addresses this gap in a single, scikit-learn-compatible
framework. Built on Venn-Abers calibration and conformal prediction, it delivers
three complementary outputs: (1) **factual rule tables** — human-readable per-instance
feature attributions with statistically valid uncertainty intervals, (2)
**counterfactual alternatives** — minimal actionable changes that would flip or shift
the prediction, and (3) **calibrated probability intervals** — empirically valid
coverage-guarantee bounds that can be used to trigger human oversight when uncertainty
is unacceptably high. These outputs map directly onto the transparency, documentation,
human oversight, accuracy, and fairness articles of the AI Act.

Deploying `calibrated_explanations` alongside an existing model requires no
architectural changes to the model itself. It wraps the model via
`WrapCalibratedExplainer`, calibrates it against a held-out calibration set, and
exposes a consistent API for generating and serialising explanation payloads. The
serialised payloads provide machine-readable audit evidence suitable for storage
in an immutable log and for submission to notified bodies or market surveillance
authorities.

---

## 2. Scope and Applicability

### 2.1 AI Act Risk Tiers Addressed

The AI Act defines four risk tiers. This document focuses on **high-risk AI systems**
as defined in **Article 6** and **Annex III**, which include systems used in:

| Annex III Area | Example use cases |
|---|---|
| 1 — Biometric identification | Identity verification, access control |
| 2 — Critical infrastructure | Grid management, transport |
| 3 — Education and vocational training | Admissions scoring, assessment |
| 4 — Employment and workers management | CV screening, performance monitoring |
| 5 — Access to essential services | Credit scoring, insurance underwriting |
| 5 — Law enforcement | Recidivism risk, threat detection |
| 6 — Migration and border control | Asylum processing, risk profiling |
| 7 — Administration of justice | Judicial decision support |

`calibrated_explanations` applies to any ML model operating in these areas. The
explanation and uncertainty capabilities described here also benefit **limited-risk**
systems subject to Art. 50 transparency obligations.

### 2.2 Supported Model Types and Tasks

| Task | CE support |
|---|---|
| Binary classification | Full — `explain_factual`, `explore_alternatives`, `predict_proba` |
| Multi-class classification | Full — per-class factual rules and alternatives |
| Regression | Full — prediction intervals, threshold-based probabilistic explanations |

Supported base estimators: any scikit-learn-compatible model (including gradient
boosting libraries such as XGBoost, LightGBM, and CatBoost via sklearn wrappers).

### 2.3 Prerequisites

1. A trained, scikit-learn-compatible model (`predict` / `predict_proba` interface).
2. A **calibration set** `(x_cal, y_cal)` — held-out data not used during training,
   recommended size ≥ 200 instances for reliable interval coverage.
3. A **proper training set** `(x_proper, y_proper)` — used only for fitting the
   internal calibration layer.
4. Python ≥ 3.9 and `calibrated_explanations` installed
   (`pip install calibrated_explanations`).

---

## 3. Article-by-Article Compliance Mapping

### Art. 9 — Risk Management System

**a) Title (verbatim):** Article 9 — Risk management system

**b) Core obligation:** Providers of high-risk AI systems must establish, implement,
document, and maintain a risk management system that identifies, analyses, and
evaluates the reasonably foreseeable risks that the system may pose, and takes
appropriate mitigation measures throughout the lifecycle (Art. 9(1)–(6)).

**c) How `calibrated_explanations` contributes:** Uncertainty quantification is a
prerequisite for rational risk management. `predict_proba` with `uq_interval=True`
provides empirically valid lower and upper probability bounds for every prediction.
Wide intervals signal cases where the model's evidence base is weak — these are
exactly the cases that pose elevated risk and that a risk management system must
identify and route to additional controls. The interval width can be used as a
quantitative risk indicator in the system's risk register.

**d) Code:**

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

# Compute prediction with uncertainty interval
probs, (low, high) = explainer.predict_proba(X_query, uq_interval=True)
interval_width = high - low

# Risk flag: interval wider than tolerance threshold
RISK_THRESHOLD = 0.30
high_risk_mask = interval_width[:, 1] > RISK_THRESHOLD
```

**e) Audit documentation:** Log `interval_width` and `high_risk_mask` per case.
Include the calibration set size and the chosen `RISK_THRESHOLD` in the AI system's
technical documentation (Annex IV, §1(c)).

---

### Art. 10 — Data and Data Governance

**a) Title (verbatim):** Article 10 — Data and data governance

**b) Core obligation:** High-risk AI systems must be trained on data that is
relevant, representative, and free from errors, and providers must take appropriate
measures to examine data for biases that could give rise to risks within the meaning
of Art. 9(2) (Art. 10(2)(f)–(g)). Where relevant, group-specific bias and accuracy
disparities must be examined and mitigated.

**c) How `calibrated_explanations` contributes:** Mondrian (conditional) calibration
via the `bins` parameter on `explain_factual`, `explore_alternatives`, or
`predict_proba` generates **group-specific prediction intervals**. By partitioning
the calibration set by protected attribute (age group, gender, nationality), the
practitioner obtains per-group coverage and interval width statistics. Wider intervals
for a particular sub-group are a direct, quantitative signal of reduced prediction
reliability for that group, satisfying the obligation to examine bias risks.

**d) Code:**

```python
import numpy as np

# group_labels: array-like of group identifiers for each calibration instance
# e.g., derived from a protected demographic attribute
group_labels_cal = x_cal[:, protected_feature_idx]
group_labels_test = X_query[:, protected_feature_idx]

# Mondrian-calibrated factual explanation — separate intervals per group
factual = explainer.explain_factual(X_query, bins=group_labels_test)

# Alternatively, at the probability level
probs, (low, high) = explainer.predict_proba(
    X_query, uq_interval=True, bins=group_labels_test
)

# Inspect per-group interval widths as bias proxy
for group in np.unique(group_labels_test):
    mask = group_labels_test == group
    width = (high[:, 1] - low[:, 1])[mask].mean()
    print(f"Group {group}: mean interval width = {width:.3f}")
```

**e) Audit documentation:** Record per-group mean interval width and coverage
estimates in the data governance section of the technical documentation. A
statistically significant disparity across groups should be flagged and the
mitigation strategy documented.

---

### Art. 11 and Annex IV — Technical Documentation

**a) Title (verbatim):** Article 11 — Technical documentation; Annex IV — Technical
documentation referred to in Article 11(1)

**b) Core obligation:** Before placing a high-risk AI system on the market,
providers must draw up technical documentation demonstrating compliance. Annex IV
specifies that this documentation must include, among other items: a description of
the system's accuracy, robustness and cybersecurity measures; the monitoring,
functioning and control mechanisms; and the measures taken to enable traceability
and auditability (Annex IV §1(c), §1(e), §2(c)).

**c) How `calibrated_explanations` contributes:** `calibrated_explanations` produces
structured, serialisable explanation objects that encode the model metadata,
calibration configuration, feature names, per-instance factual rules, and uncertainty
intervals. The JSON payload produced by `as_json()` provides a machine-readable
record that satisfies the traceability requirement. The calibration configuration
(calibration set size, conformal significance level, whether Mondrian binning is
used) should be included verbatim in the Annex IV technical documentation.

**d) Code:**

```python
# Generate explanation and serialise as JSON for technical documentation
explanation = explainer.explain_factual(X_query)

# JSON payload — suitable for archiving in immutable audit log
payload = explanation[0].as_json()  # single-instance payload
print(payload)
# Contains: feature names, values, weights, intervals, prediction, probability

# For a multi-row batch, iterate
import json, pathlib
log_path = pathlib.Path("audit_logs") / "explanations.jsonl"
log_path.parent.mkdir(exist_ok=True)
with log_path.open("a") as f:
    for exp in explanation:
        f.write(json.dumps(exp.as_json()) + "\n")
```

**e) Audit documentation:** The technical documentation must reference the
`calibrated_explanations` version (retrievable via
`import calibrated_explanations; calibrated_explanations.__version__`), the
calibration set provenance, and the significance level (confidence) used. Store
the JSON schema of the explanation payload as Annex IV §3 supporting artefact.

---

### Art. 12 — Record-keeping and Logging

**a) Title (verbatim):** Article 12 — Record-keeping

**b) Core obligation:** High-risk AI systems must automatically log events relevant
to the system's operation, including dates and times of use, reference to the input
data, and the results of the AI system's operation, for a period appropriate to the
intended purpose (Art. 12(1)–(2)).

**c) How `calibrated_explanations` contributes:** `as_json()` produces a complete,
self-contained record of every prediction event, including the input vector, the
predicted class or value, the calibrated probability, the uncertainty interval, and
the factual feature-weight rule table. Writing this payload to an append-only log
(JSONL file, database, or audit service) satisfies the logging obligation without
requiring a separate logging infrastructure.

**d) Code:**

```python
import json, datetime, pathlib, uuid

audit_log = pathlib.Path("audit_logs") / "ai_act_record.jsonl"
audit_log.parent.mkdir(exist_ok=True)

explanation = explainer.explain_factual(X_query)

with audit_log.open("a", encoding="utf-8") as f:
    for i, exp in enumerate(explanation):
        record = {
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "model_version": model_version,
            "ce_version": calibrated_explanations.__version__,
            "input_index": i,
            "explanation": exp.as_json(),
        }
        f.write(json.dumps(record) + "\n")
```

**e) Audit documentation:** Define and document the log retention policy (Art. 12(2)
specifies a period proportionate to the purpose; credit decisions commonly require
≥ 3 years). Include the log schema and access-control policy in the AI system's
technical documentation.

---

### Art. 13 — Transparency and Provision of Information to Deployers

**a) Title (verbatim):** Article 13 — Transparency and provision of information to
deployers

**b) Core obligation:** High-risk AI systems must be designed and developed to
ensure that their operation is sufficiently transparent to enable deployers to
interpret the system's outputs and use them appropriately (Art. 13(1)). The
instructions for use must include information about the system's capabilities and
limitations (Art. 13(3)(b)).

**c) How `calibrated_explanations` contributes:** `explain_factual` returns a
human-readable **factual rule table** for each instance: a ranked list of features
together with their actual values, the direction and magnitude of their contribution
to the prediction, and the uncertainty interval around each weight. This output is
directly interpretable by a non-technical deployer — the rule reads, in natural
language, as "because feature X had value V, the probability of outcome Y increased
by W [±Δ]". The rule table constitutes the "sufficiently transparent" output
required by Art. 13(1).

**d) Code:**

```python
# Per-instance factual explanation
factual = explainer.explain_factual(X_query)

# Print human-readable rule table for the first instance
factual[0].print_rules()
# Example output:
#   age = 42         →  P(approve) increased by 0.12 [0.07, 0.18]
#   income = 55000   →  P(approve) increased by 0.09 [0.04, 0.15]
#   debt_ratio = 0.4 →  P(approve) decreased by 0.05 [0.01, 0.10]

# Access structured rule data programmatically
rules = factual[0].as_json()
```

**e) Audit documentation:** Log the full rule table for every production prediction
event. The instructions for use (Art. 13(3)) should describe the rule-table format
and explain how deployers should interpret uncertainty intervals, including the
guidance that predictions with wide intervals require additional review.

---

### Art. 14 — Human Oversight

**a) Title (verbatim):** Article 14 — Human oversight

**b) Core obligation:** High-risk AI systems must be designed and developed to
allow the natural persons to whom human oversight is assigned to effectively
oversee the system's operation. The system must enable those persons to decide not
to use the system or to override, disregard, or reverse its output (Art. 14(1),
(4)(c)).

**c) How `calibrated_explanations` contributes:** The **interval straddle policy**
and **interval width policy** (both built-in decision-policy patterns) provide
automatic, principled triggers for routing predictions to human review. When the
calibrated probability interval straddles the decision boundary, or when the
interval width exceeds a pre-defined tolerance, the system flags the case for human
oversight *before* any automated decision is taken. This is not a post-hoc safeguard;
it is an inline gate that prevents the automated system from acting on uncertain
predictions without human confirmation.

**d) Code:**

```python
from calibrated_explanations.core.reject.policy import RejectPolicy

probs, (low, high) = explainer.predict_proba(X_query, uq_interval=True)
decision_boundary = 0.5

# Straddle check: probability interval crosses the decision boundary
straddles = (low[:, 1] < decision_boundary) & (high[:, 1] > decision_boundary)

# Width check: absolute uncertainty too large
MAX_WIDTH = 0.25
too_uncertain = (high[:, 1] - low[:, 1]) > MAX_WIDTH

# Combined human-oversight gate
needs_human_review = straddles | too_uncertain

for idx in X_query[needs_human_review]:
    escalate_to_human_reviewer(idx, reason="uncertainty_gate")

# Alternatively, use built-in RejectPolicy via explain_factual
factual = explainer.explain_factual(X_query, reject_policy=RejectPolicy.FLAG)
```

**e) Audit documentation:** Log every escalation event with the case identifier,
the triggering condition (`straddle`, `width`, or both), the interval values, and
the human reviewer's identity and decision. This log constitutes the human oversight
audit trail required by Art. 14 and Recital 58.

---

### Art. 15 — Accuracy, Robustness and Cybersecurity

**a) Title (verbatim):** Article 15 — Accuracy, robustness and cybersecurity

**b) Core obligation:** High-risk AI systems must achieve an appropriate level of
accuracy, robustness, and consistency of performance. Accuracy metrics must be
declared in the technical documentation. Providers must document the expected level
of accuracy and the measures taken to achieve it (Art. 15(1)–(2)).

**c) How `calibrated_explanations` contributes:** Venn-Abers calibration and
conformal prediction deliver **empirical coverage guarantees**: the stated significance
level (e.g., 90 % confidence) corresponds to a statistically verifiable empirical
coverage on the calibration set. This is a stronger claim than a point accuracy
metric — it is a **certifiable bound** on prediction interval validity. The
calibration error (difference between nominal and empirical coverage) can be measured
and reported as a quantitative accuracy declaration for Annex IV.

**d) Code:**

```python
# Measure empirical coverage on a held-out validation set
probs, (low, high) = explainer.predict_proba(x_val, uq_interval=True)
nominal_confidence = 0.90  # passed as `confidence` at calibrate() time

# For classification: empirical coverage = fraction of true labels within interval
in_interval = (low[range(len(y_val)), y_val] <= probs[range(len(y_val)), y_val]) & \
              (probs[range(len(y_val)), y_val] <= high[range(len(y_val)), y_val])
empirical_coverage = in_interval.mean()

print(f"Nominal confidence: {nominal_confidence:.2%}")
print(f"Empirical coverage: {empirical_coverage:.2%}")
# A well-calibrated system: empirical ≈ nominal (difference < 2 pp)
```

**e) Audit documentation:** Report both nominal confidence and empirical coverage
in the technical documentation's accuracy section. Repeat this measurement for each
sub-population identified in the data governance section (Art. 10), and for each
model version deployed. Include the calibration set size and split procedure
(stratified vs. random) as Annex IV §1(c) artefacts.

---

### Art. 50 — Transparency Obligations for Certain AI Systems

**a) Title (verbatim):** Article 50 — Transparency obligations for providers and
deployers of certain AI systems

**b) Core obligation:** Where an AI system is used to make or assist in making
decisions that significantly affect persons, those persons must be informed of
their right to receive an explanation of the decision (Art. 50(1)). The explanation
must be meaningful, intelligible, and in plain language (Recital 47).

**c) How `calibrated_explanations` contributes:** `explore_alternatives` generates
**counterfactual (alternative) explanations**: the minimal set of feature changes
that would produce a different or more favourable prediction. These are directly
usable as the Art. 50 "explanation to the individual" — they communicate not only
*why* the current decision was made but *what the individual could change* to obtain
a different outcome. The output is inherently actionable, non-discriminatory (it
proposes changes only to features that are within the individual's control, when
properly configured), and expressible in plain language.

**d) Code:**

```python
# Generate counterfactual alternatives for the affected individual
alternatives = explainer.explore_alternatives(X_query)

# Print actionable alternatives for person at index 0
alternatives[0].print_rules()
# Example output:
#   IF income increased from 38000 → 52000 THEN P(approve) = 0.73 [0.65, 0.81]
#   IF debt_ratio decreased from 0.55 → 0.35 THEN P(approve) = 0.68 [0.59, 0.76]

# Serialise for delivery to the affected person / inclusion in decision letter
alt_json = alternatives[0].as_json()
```

**e) Audit documentation:** Whenever a significant automated or assisted decision
is taken, log the full `explore_alternatives` output for the affected individual.
This constitutes the "explanation record" that must be made available on request
(Art. 50, Recital 47). The log entry must include the instance identifier, the
final decision, the probability and interval, and the alternative rules.

---

## 4. Compliance Checklist

| Article | Obligation summary | CE feature / method | Status |
|---|---|---|---|
| Art. 9 | Identify and quantify AI system risks throughout lifecycle | `predict_proba(uq_interval=True)` — interval width as risk indicator | **Covered** |
| Art. 10 | Examine training data for bias; ensure representativeness across groups | `explain_factual(bins=group_labels)` — Mondrian per-group intervals | **Covered** |
| Art. 11 + Annex IV | Produce technical documentation including accuracy and traceability | `as_json()` — serialisable explanation payload; calibration configuration | **Covered** |
| Art. 12 | Log events: input data, result, date/time — for defined retention period | `as_json()` appended to audit log with timestamp and record ID | **Partially covered** — log infrastructure and retention policy must be provided by deployer |
| Art. 13 | Transparent operation; deployer can interpret system outputs | `explain_factual()` — `print_rules()` human-readable rule table | **Covered** |
| Art. 14 | Enable human oversight; allow override before automated action | `predict_proba` straddle/width gates; `RejectPolicy.FLAG` | **Covered** |
| Art. 15 | Declared, verifiable accuracy; robustness documentation | Empirical coverage vs. nominal confidence (Venn-Abers guarantee) | **Covered** |
| Art. 50 | Right to explanation; inform affected persons in plain language | `explore_alternatives()` — actionable counterfactual rules | **Covered** |

---

## 5. Limitations and Gaps

The following compliance obligations are **not** satisfied by `calibrated_explanations`
alone and require additional organisational or technical controls.

### 5.1 Data Quality Certification (Art. 10(2)(a)–(e))
Art. 10 requires that training data satisfies relevance, representativeness,
error-freedom, and completeness criteria. `calibrated_explanations` does not
inspect, validate, or certify input data quality. **Required:** A data quality
framework (e.g., Great Expectations, Soda, or a custom data validation pipeline)
upstream of model training, with documented data quality test results.

### 5.2 Cybersecurity Controls (Art. 15(3)–(5))
Art. 15 requires resilience against adversarial manipulation, data poisoning, and
model-evasion attacks. `calibrated_explanations` does not provide adversarial
robustness defences. **Required:** Adversarial testing (e.g., using libraries such
as `adversarial-robustness-toolbox`) and network/access controls protecting the
model serving endpoint.

### 5.3 Conformity Assessment and CE Marking (Art. 43–49)
High-risk AI systems must undergo a conformity assessment before market placement.
`calibrated_explanations` provides technical evidence for the assessment but is not
itself a conformity assessment tool. **Required:** Engagement with a notified body
(where applicable) or internal conformity assessment process, documentation in
accordance with Art. 11 and Annex IV, and registration in the EU AI Act database
(Art. 49).

### 5.4 Post-Market Monitoring (Art. 72)
Art. 72 requires providers to operate a post-market monitoring system that
continuously collects data on system performance after deployment. `calibrated_explanations`
does not provide drift detection, performance dashboards, or monitoring alerts.
**Required:** A model monitoring system (e.g., Evidently AI, WhyLabs, or a custom
pipeline) that tracks empirical coverage, interval widths, and prediction
distributions over time, and feeds back into the risk management system (Art. 9).

### 5.5 Human Oversight Processes Outside the Model (Art. 14(4)(a)–(d))
Art. 14 requires that humans assigned to oversight are sufficiently competent to
understand the system's capabilities and limitations, and that their decisions are
logged. `calibrated_explanations` can flag cases for review but cannot train
reviewers, enforce review quality, or impose a review workflow. **Required:**
Documented escalation procedures, reviewer training, and a workflow management
system that records reviewer decisions and links them back to the CE audit log.

### 5.6 Fundamental Rights Impact Assessment (Art. 9(9), Recital 66)
For systems that may significantly impact fundamental rights, a fundamental rights
impact assessment is required. This is an organisational and legal exercise.
**Required:** A documented FRIA process, conducted with legal counsel and (where
relevant) a Data Protection Officer, referencing the per-group bias analysis enabled
by Mondrian calibration as quantitative evidence.

---

## 6. Quick-Start Integration Guide

The following recipe delivers a compliance-ready deployment in six steps. Each
step is annotated with the AI Act article it primarily satisfies.

### Step 1 — Install

```bash
pip install calibrated_explanations
```

### Step 2 — Wrap and Calibrate the Model

*(Prerequisite: `x_proper`, `y_proper` = proper training set; `x_cal`, `y_cal` =
held-out calibration set; `feature_names` = list of feature name strings)*

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)
```

### Step 3 — Art. 13: Generate a Factual Explanation (Transparency)

```python
# X_query: the feature vector(s) of the instance(s) to be decided upon
factual = explainer.explain_factual(X_query)

# Human-readable rule table — include in deployer UI and decision letter
factual[0].print_rules()

# Structured payload for logging and documentation
payload = factual[0].as_json()
```

### Step 4 — Art. 50: Retrieve Counterfactual Alternatives

```python
alternatives = explainer.explore_alternatives(X_query)

# Print actionable alternatives — include in explanation provided to individual
alternatives[0].print_rules()

alt_payload = alternatives[0].as_json()
```

### Step 5 — Art. 14: Apply a Reject / Escalation Policy (Human Oversight)

```python
probs, (low, high) = explainer.predict_proba(X_query, uq_interval=True)
decision_boundary = 0.5
MAX_INTERVAL_WIDTH = 0.25

straddles = (low[:, 1] < decision_boundary) & (high[:, 1] > decision_boundary)
too_uncertain = (high[:, 1] - low[:, 1]) > MAX_INTERVAL_WIDTH
needs_review = straddles | too_uncertain

if needs_review.any():
    escalate_to_human_reviewer(X_query[needs_review], reason="uncertainty_gate")
```

### Step 6 — Art. 12: Serialise the Audit Payload (Record-keeping)

```python
import json, datetime, pathlib, uuid
import calibrated_explanations

audit_log = pathlib.Path("audit_logs") / "ai_act_record.jsonl"
audit_log.parent.mkdir(exist_ok=True)

with audit_log.open("a", encoding="utf-8") as f:
    for i, (fact, alt) in enumerate(zip(factual, alternatives)):
        record = {
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "ce_version": calibrated_explanations.__version__,
            "factual": fact.as_json(),
            "alternatives": alt.as_json(),
            "human_review_required": bool(needs_review[i]),
        }
        f.write(json.dumps(record) + "\n")
```

---

## 7. References

### Regulatory

- **EU AI Act** — Regulation (EU) 2024/1689 of the European Parliament and of the
  Council of 13 June 2024 laying down harmonised rules on artificial intelligence.
  Official Journal of the European Union, L series, 12 July 2024.
  <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689>
- **Relevant recitals:** Recital 47 (meaningful explanation), Recital 48 (right to
  explanation does not apply to purely automatic decisions under GDPR unless
  separately required), Recital 58 (human oversight design), Recital 66
  (fundamental rights impact), Recital 71 (accuracy and performance metrics).
- **Annex III** — List of high-risk AI systems referred to in Art. 6(2).
- **Annex IV** — Technical documentation referred to in Art. 11(1).

### Calibrated Explanations

- Löfström, H., Löfström, T., Johansson, U., & Sönströd, C. (2024).
  *Calibrated Explanations: with Uncertainty Information and Counterfactuals.*
  Expert Systems with Applications. DOI: [10.1016/j.eswa.2024.123154](https://doi.org/10.1016/j.eswa.2024.123154)
- Löfström, T., Löfström, H., Johansson, U., Sönströd, C., & Matela, R. (2024).
  *Calibrated Explanations for Regression.*
  Machine Learning. DOI: [10.1007/s10994-024-06642-8](https://doi.org/10.1007/s10994-024-06642-8)
- Löfström, H., Löfström, T., Johansson, U., Sönströd, C., & Boström, H. (2024).
  *Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty.*
  In: Proc. ECAI 2024 Workshop. DOI: [10.1007/978-3-031-63787-2_17](https://doi.org/10.1007/978-3-031-63787-2_17)
- Repository: <https://github.com/Moffran/calibrated_explanations>

### Conformal Prediction and Venn-Abers

- Shafer, G., & Vovk, V. (2008).
  *A Tutorial on Conformal Prediction.*
  Journal of Machine Learning Research, 9, 371–421.
- Vovk, V., Gammerman, A., & Shafer, G. (2005).
  *Algorithmic Learning in a Random World.* Springer.
- Johansson, U., Löfström, T., & Boström, H. (2021).
  *Venn-Abers Predictors.*
  In: Conformal and Probabilistic Prediction with Applications (COPA 2021).
