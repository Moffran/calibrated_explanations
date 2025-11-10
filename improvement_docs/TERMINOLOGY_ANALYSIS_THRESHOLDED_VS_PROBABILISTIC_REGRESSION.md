# In-Depth Terminology Analysis: "Thresholded Regression" vs "Probabilistic Regression"

**Analysis Date:** November 9, 2025
**Status:** Comprehensive analysis of terminology usage and definitions across documentation and codebase
**Scope:** All references in documentation (`.md` files), ADRs, design documents, and Python code

---

## Executive Summary

**Finding:** The terms **"thresholded regression"** and **"probabilistic regression"** refer to **the same concept**, but this equivalence is **poorly documented** and **inconsistently applied** across the codebase.

### Key Observations

1. **No explicit definition** exists that formally states these terms are equivalent
2. **ADR-021** uses **"thresholded regression"** almost exclusively to describe the technical implementation
3. **User-facing documentation** (quickstarts, guides, README) prefers **"probabilistic regression"** for clarity and marketability
4. **Implementation code** uses both terms inconsistently in comments and metadata
5. **No test adequately documents this equivalence**
### Recommendation

**Adopt "Probabilistic Regression" as the canonical term** for user-facing documentation and public APIs. Reserve "thresholded regression" for technical architecture discussions (ADRs, internal design docs, technical implementation comments). This provides both marketing clarity and technical precision.

---

## Section 1: Definition Analysis

### 1.1 What Does "Probabilistic Regression" Mean?

**Definition from:** `docs/foundations/concepts/probabilistic_regression.md` (lines 1-3)

> "Probabilistic regression extends calibrated explanations beyond point estimates. It pairs calibrated probabilities ("what is the chance the outcome exceeds my threshold?") with calibrated intervals that describe where the numeric target is likely to fall."

**Core capability:**
- Takes a regression model's continuous predictions
- Applies a user-defined **threshold** value
- Returns calibrated **probabilities** for whether the target is above/below (or within) the threshold
- Produces calibrated **interval bounds** around those probabilities

**User-level API signature:**
```python
probabilities, probability_interval = explainer.predict_proba(
    X_test[:1],
    threshold=150,      # The "probabilistic" aspect: threshold probability
    uq_interval=True,   # Return interval bounds for uncertainty quantification
)
```

### 1.2 What Does "Thresholded Regression" Mean?

**Definition from:** `improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md` (lines 23-24, 83-104)

> "Thresholded regression returns calibrated probabilities for threshold events (`y \leq t` or `t_0 < y \leq t_1`) and is the only regression path that relies on both conformal predictive systems (CPS) *and* Venn-Abers."

**Technical description:**
- The regression model's predictions are **thresholded** (converted into binary classification targets)
- Calibration uses a **hybrid approach**:
  - CPS (Conformal Predictive Systems) to produce base probabilities
  - Venn-Abers to calibrate those probabilities
- Returns probability predictions and probability intervals

**Implementation code path from `IntervalRegressor.predict_probability()`:**
```python
# Converts regression predictions to probabilities by thresholding
proba = self.split["cps"].predict(y_hat=..., y=y_threshold, ...)
# Then calibrates with Venn-Abers
va = VennAbers(None, (self.ce.y_cal[cal_va] <= y_threshold).astype(int), ...)
```

### 1.3 Evidence of Equivalence

#### 1.3.1 ADR-021 Lines 195 and 204 (SEMANTIC EQUIVALENCE)

From `improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md`:

```markdown
### 3. Thresholded regression: CPS probabilities calibrated by Venn-Abers

* Thresholded regression keeps the `IntervalRegressor` but activates its
  probabilistic path. The explainer first validates and normalizes the
  threshold argument (scalar, tuple, or per-instance sequence)...
```

**Note:** The term "probabilistic path" is used to describe the "thresholded regression" implementation, indicating these refer to the same technical execution path.

#### 1.3.2 DESIGN.md Lines 21 (EXPLICIT TREATMENT)

From `external_plugins/shap_lime/DESIGN.md`:

> "Probabilistic regression requests (non-`None` threshold) are treated identically to classification because the calibrated outputs already represent event probabilities as mandated by ADR-021."

**Interpretation:** "Probabilistic regression requests" = regression predictions with a threshold argument (i.e., "thresholded regression").

#### 1.3.3 Code Comments (INDIRECT EQUIVALENCE)

From `src/calibrated_explanations/core/calibrated_explainer.py` line 3243:

```python
return (
    prediction - instance_predict
    if np.isscalar(prediction)
    else [prediction[i] - ip for i, ip in enumerate(instance_predict)]
)  # probabilistic regression
```

**Context:** This code appears in the `_assign_weight` method for both classification and probabilistic regression, indicating the two branches converge at this point.

### 1.4 Semantic Distinction (Why the Two Names?)

Both terms describe **regression with a threshold**, but emphasize different aspects:

| Aspect | "Thresholded Regression" | "Probabilistic Regression" |
|--------|--------------------------|---------------------------|
| **Emphasis** | Technical: the threshold operation | User outcome: probability prediction |
| **Audience** | Architecture, technical docs, implementers | Practitioners, researchers, marketing |
| **Context** | Implementation details, internal calibration flows | User APIs, quickstarts, papers |
| **Formality** | More technical, precise | More accessible, end-user-focused |

**Neither term is "wrong"—they are complementary framings of the same feature.**

---

## Section 2: Comprehensive Terminology Inventory

### 2.1 Documentation Usage: "Probabilistic Regression"

#### Found In:

| File | Context | References |
|------|---------|-----------|
| `README.md:118` | Quickstart link | "probabilistic regression quickstart" |
| `README.md:159` | Feature highlight | "Probabilistic and interval regression that mirrors the classification API" |
| `docs/get-started/index.md:7-9` | Main navigation | "Probabilistic regression" (paired with classification) |
| `docs/get-started/quickstart_regression.md:81` | Link to concept guide | Points to probabilistic regression guide |
| `docs/foundations/concepts/probabilistic_regression.md` | **Dedicated page** (entire file) | Complete user guide with examples and research links |
| `docs/get-started/index.md:25-26` | Feature table | "probabilistic thresholds pair with interval regression" |
| `docs/citing.md:76, 135` | Citation abstracts | "probabilistic regression", "thresholded regression tasks" |
| `docs/researcher/advanced/theory_and_literature.md:18, 65` | Research page | Links to journal articles on "probabilistic regression" |
| `notebooks/demo_probabilistic_regression.ipynb` | **Example notebook** | Complete workflow demonstration |
| `external_plugins/test_calibrated_lime_shap.py:3, 134-135` | Plugin test | "Passing a threshold argument triggers probabilistic regression behaviour" |

#### Total: **~35 documentation references** to "probabilistic regression"

### 2.2 Documentation Usage: "Thresholded Regression"

#### Found In:

| File | Context | References |
|------|---------|-----------|
| `perturbation2.md:99` | Experiment description | "thresholded regression task (turn regression score into a decision)" |
| `perturbation3.md:195` | Results summary | "regression/thresholded regression (CPS)" |
| `perturbation_evaluation.md:44, 67` | Evaluation metrics | "Regression / thresholded regression oracle" |
| `improvement_docs/adrs/ADR-021:23, 83, 85, 167, 183, 185` | **Core ADR** (multiple sections) | Technical architecture using "thresholded regression" consistently |
| `improvement_docs/archived/ADR-analysis.md:1343, 1350` | ADR review | "percentile regression, and thresholded regression" |
| `improvement_docs/legacy_user_api_contract.md:36, 173` | Legacy API | "thresholded regression predictions for probabilistic labels" |
| `improvement_docs/LEGACY_TO_PLOTSPEC_MAPPING.md:104` | Plotting metadata | "is_probabilistic: boolean (True when classification or thresholded regression)" |
| `docs/foundations/governance/optional_telemetry.md:66` | Telemetry docs | "thresholded regression runs remain auditable" |

#### Total: **~22 documentation references** to "thresholded regression"

### 2.3 Code Usage Patterns

#### 2.3.1 Python Parameter Names and Signatures

| Location | Parameter | Usage |
|----------|-----------|-------|
| `src/calibrated_explanations/api/config.py:35, 100` | `threshold: float \| None` | "for probabilistic regression use-cases" |
| `src/calibrated_explanations/core/interval_regressor.py:136, 158` | `y_threshold` parameter | Method `predict_probability(x, y_threshold, bins=None)` |
| `src/calibrated_explanations/explanations/explanations.py` | `y_threshold` attribute | Stores threshold for probabilistic explanations |
| `src/calibrated_explanations/core/calibrated_explainer.py:1973, 1980` | `threshold` parameter | `_predict_impl(x, threshold=None, ...)` |

#### 2.3.2 Method Names

| Location | Method | Context |
|----------|--------|---------|
| `src/calibrated_explanations/explanations/explanations.py:395` | `_is_thresholded()` | Property check in `CalibratedExplanations` |
| `tests/plugins/test_builtins_behaviour.py:400, 447, 499, 545, 592` | `is_thresholded()` | Mock explainer property |
| `tests/unit/test_explanations_collection.py:262` | `_is_thresholded()` | Test assertion |

#### 2.3.3 Comments and Docstrings

| Location | Text | Usage |
|----------|------|-------|
| `src/calibrated_explanations/core/interval_regressor.py:136-155` | Docstring | "Predict the probabilities for each instance...being above the threshold(s)" |
| `src/calibrated_explanations/core/calibrated_explainer.py:1987-2000` | Docstring | "For regression: Returns predictions and uncertainty intervals. Can return probability predictions when threshold is provided" |
| `improvement_docs/adrs/ADR-021:83-104` | Section title | "### 3. Thresholded regression: CPS probabilities calibrated by Venn-Abers" |

### 2.4 Test Names and Test Coverage

#### Test Files Using "Probabilistic Regression":

```python
# From tests/integration/core/test_regression.py
def test_probabilistic_regression_ce(regression_dataset)
def test_probabilistic_regression_int_threshold_ce(regression_dataset)
def test_probabilistic_regression_conditional_ce(regression_dataset)
def test_probabilistic_regression_conditional_fast_ce(regression_dataset)
def test_knn_normalized_probabilistic_regression_ce(regression_dataset)
def test_var_normalized_probabilistic_regression_ce(regression_dataset)
def test_probabilistic_regression_fast_ce(regression_dataset)
```

**Count:** ~7-10 test names with "probabilistic_regression"

#### Test Files Using "Thresholded Regression":

```python

def test_probabilistic_regression_threshold(self):
```

**Count:** ~2 test names; "thresholded" appears in docstring rather than name

### 2.5 Notebooks

| File | Content |
|------|---------|
| `notebooks/demo_probabilistic_regression.ipynb` | **Entire notebook** dedicated to probabilistic regression workflow |
| `external_plugins/save_calibrated_plugins_demo.ipynb:9, 379-380` | "Probabilistic regression example (SHAP)" |

---

## Section 3: Analysis of How Terms Are Used in Different Contexts

### 3.1 User-Facing Documentation (Public APIs, Quickstarts, Guides)

**Preferred Term:** **"Probabilistic Regression"** (consistent)

**Examples:**
- `docs/get-started/quickstart_regression.md` – Uses "probabilistic regression" in user examples
- `README.md` – Links to "probabilistic regression quickstart"
- `docs/foundations/concepts/probabilistic_regression.md` – Entire concept guide titled with this term
- `notebooks/demo_probabilistic_regression.ipynb` – Notebook uses this term

**Rationale:** Clearer to practitioners; emphasizes the probability output rather than the technical operation. Differentiates from "percentile regression" (interval-only).

### 3.2 Technical Architecture & Implementation (ADRs, Design Docs)

**Preferred Term:** **"Thresholded Regression"** (inconsistently applied)

**Examples:**
- `ADR-021-calibrated-interval-semantics.md` – Uses "thresholded regression" in section 3 to describe the technical pathway
- `improvement_docs/adrs/ADR-013` – References "thresholded regression" in comments about CPS/Venn-Abers composition
- `external_plugins/shap_lime/DESIGN.md:21` – States "Probabilistic regression requests (non-None threshold)" = same meaning

**Rationale:** More precise for implementation details; describes the specific operation (threshold) and calibration strategy. Technical readers understand the operation vs. the user outcome.

### 3.3 Evaluation & Benchmarking

**Used in:** `perturbation2.md`, `perturbation3.md`, `perturbation_evaluation.md`

**Terminology:** Exclusively **"thresholded regression"**

**Context:** Research evaluation docs describe the benchmarking setup. Uses "thresholded regression" as a contrast to "percentile regression."

---

## Section 4: Current Gaps in Documentation

### 4.1 Gap #1: No Formal Definition Statement

**Problem:** Nowhere in the repository is it explicitly stated:

> "Thresholded regression and probabilistic regression refer to the same capability."

**Where a definition should exist:**
1. In `ADR-021` (would clarify terminology for implementers)
2. In `docs/foundations/concepts/probabilistic_regression.md` (could note "also called thresholded regression")
3. In a new "Terminology" section of contributor docs

**Impact:** New contributors may be confused about whether these are different features. Users reading both ADRs and quickstarts encounter inconsistent terminology without clear mapping.

### 4.2 Gap #2: Test Docstrings Don't Clarify Equivalence

**Problem:** Test docstrings mix terminology inconsistently:

```python
def test_lime_plugin_probabilistic_regression():
    """Test thresholded regression task."""  # Contradiction!
```

**Finding:** The function name uses "probabilistic" but the docstring uses "thresholded." This confuses readers about whether they test the same thing.

### 4.3 Gap #3: Method Names Vary Without Explanation

**Problem:** The codebase uses both `y_threshold` and `threshold` parameters, and `_is_thresholded()` method, but never explains why "thresholded" is used in code while users interact with "probabilistic regression."

### 4.4 Gap #4: No Glossary or Migration Path

**Problem:** If a decision is made to standardize on one term, there's no existing structure to guide the rollout across:
- Test names
- API docs
- Docstrings
- Comments
- Internal parameter names

---

## Section 5: Pros and Cons of Each Term

### 5.1 "Probabilistic Regression"

#### Pros:
1. **User-friendly:** Immediately conveys the output type (probabilities)
2. **Parallels classification:** "Probabilistic classification" is well-known; this mirrors it
3. **Already dominant in user docs:** Quickstarts, notebooks, and concept guides use this consistently
4. **Differentiates from percentile regression:** Users understand the distinction
5. **Searchable:** More likely to be found by practitioners Googling "regression probabilities"
6. **Marketing advantage:** Emphasizes novel capability (probabilities from regression models)
7. **Established in papers:** The research publications use "probabilistic regression"

#### Cons:
1. **Less precise technically:** Doesn't explicitly state the mechanism (threshold operation)
2. **Implementation detail obscured:** Code reviewers may not immediately understand the thresholding step
3. **Could be confused with Bayesian probabilistic regression:** Although context usually clarifies

### 5.2 "Thresholded Regression"

#### Pros:
1. **Technically precise:** Clearly describes what operation is applied (thresholding)
2. **Differentiates from percentile regression:** `threshold=None` → percentile; `threshold=value` → thresholded
3. **Matches code patterns:** Already used in parameter names, method logic
4. **Clear to implementers:** The internal mechanisms (CPS → threshold event → Venn-Abers) are clarified
5. **Avoids confusion with Bayesian methods:** Explicitly non-Bayesian

#### Cons:
1. **Not user-friendly:** Users don't think in terms of "thresholding a regression model"
2. **Marketing disadvantage:** Doesn't highlight the novel capability
3. **Requires explanation:** Each user doc must explain what "thresholded" means
4. **Inconsistent with classification naming:** Classification doesn't have a "thresholded" branch (it's just binary classification)
5. **Underused in papers:** Research publications prefer "probabilistic regression"

---

## Section 6: Recommendation

### Recommended Decision: **Standardize on "Probabilistic Regression"**

#### Rationale:

1. **Alignment with user-facing documentation:** The current trajectory already favors this term in README, quickstarts, guides, and notebooks.

2. **Alignment with research:** Published papers use "probabilistic regression" (see `docs/citing.md`).

3. **User clarity:** Practitioners immediately understand they're getting probability predictions from a regression model.

4. **Consistency with classification:** Mirrors the naming of "probabilistic classification" → encourages parallel workflows.

5. **Practical precedent:** The dedicated concept guide, notebook, and quickstarts are already named with this term. Changing would require massive documentation rewrites.

#### Implementation Strategy:

**Tier 1: User-Facing (Immediate)**
- ✅ Already correct in: `README.md`, quickstarts, concept guides, notebooks
- No changes required

**Tier 2: Internal Technical Documentation (v0.9.1 release window)**
- Update `ADR-021` to include a "Terminology" section stating:
  ```markdown
  ### Terminology: "Probabilistic Regression" vs. "Thresholded Regression"

  These terms are synonymous. "Probabilistic regression" is the user-facing term
  emphasizing the output (calibrated probabilities). "Thresholded regression" is
  the technical term describing the implementation (a threshold operation converts
  regression predictions into a binary event, calibrated by CPS + Venn-Abers).
  ```
- Add cross-references in `ADR-013` (interval plugin strategy)

**Tier 3: Code Changes (v0.9.1+ implementation phase)**
- Rename internal `_is_thresholded()` → `_is_probabilistic_regression()` for consistency
- Keep parameter names `threshold` and `y_threshold` (these describe the value, not the mode)
- Update method docstrings to use "probabilistic regression" while mentioning the threshold mechanism
- Update all inline comments to prefer "probabilistic regression" with technical clarifications

**Tier 4: Test Updates (v0.9.1)**
- Rename test functions to use "probabilistic_regression" consistently
  - `test_lime_plugin_probabilistic_regression()` docstring update: "Test probabilistic regression (thresholded) task."
- Ensure docstrings clarify both terms

**Tier 5: Changelog (v0.9.1 release notes)**
- Document the terminology standardization:
  ```
  ## Terminology Standardization
  - Standardized on "probabilistic regression" as the canonical term for regression
    with threshold-based probability predictions. "Thresholded regression" remains
    used in technical documents (ADRs, architecture discussions) to describe the
    implementation mechanism (CPS-based threshold calibration).
  ```

---

## Section 7: Comprehensive Reference Tables

### 7.1 All "Probabilistic Regression" References

| File | Line(s) | Context | Type |
|------|---------|---------|------|
| `README.md` | 118, 159 | Quickstart link, feature | User doc |
| `docs/get-started/index.md` | 7, 9, 25 | Navigation, feature table | User doc |
| `docs/get-started/quickstart_regression.md` | 81 | Link to concept guide | User doc |
| `docs/foundations/concepts/probabilistic_regression.md` | **entire file** | Dedicated concept guide | User doc |
| `docs/researcher/advanced/theory_and_literature.md` | 18, 65 | Research citations | User doc |
| `docs/citing.md` | 76, 135 | Paper abstracts | User doc |
| `notebooks/demo_probabilistic_regression.ipynb` | **entire file** | Example notebook | Code |
| `external_plugins/test_calibrated_lime_shap.py` | 3, 134–135 | Test comments | Code |
| `external_plugins/save_calibrated_plugins_demo.ipynb` | 9, 379–380 | Example notebook | Code |
| `external_plugins/shap_lime/DESIGN.md` | 21 | Design document | Tech doc |
| `external_plugins/reject/DESIGN.md` | 12, 20 | Design document | Tech doc |
| `src/calibrated_explanations/api/config.py` | 35, 100 | Parameter docstring | Code |
| `src/calibrated_explanations/core/calibrated_explainer.py` | 3243 | Comment | Code |
| `tests/integration/core/test_regression.py` | **~10 test functions** | Test names | Test |
| `improvement_docs/legacy_user_api_contract.md` | 127, 138 | API examples | Tech doc |
| `improvement_docs/documentation_*.md` | **multiple** | Doc architecture | Tech doc |
| `improvement_docs/RELEASE_PLAN_v1.md` | **multiple** | Release planning | Tech doc |

**Total: ~60 references**

### 7.2 All "Thresholded Regression" References

| File | Line(s) | Context | Type |
|------|---------|---------|------|
| `perturbation2.md` | 99 | Experiment description | Research doc |
| `perturbation3.md` | 195 | Results summary | Research doc |
| `perturbation_evaluation.md` | 44, 67 | Metrics | Research doc |
| `improvement_docs/adrs/ADR-021` | 23, 83–104, 167, 183, 185 | **Core architecture** | ADR |
| `improvement_docs/archived/ADR-analysis.md` | 1343, 1350 | ADR review | Tech doc |
| `improvement_docs/legacy_user_api_contract.md` | 36, 173 | Legacy API | Tech doc |
| `improvement_docs/LEGACY_TO_PLOTSPEC_MAPPING.md` | 104 | Plotting metadata | Tech doc |
| `docs/foundations/governance/optional_telemetry.md` | 66 | Telemetry docs | Tech doc |
| `src/calibrated_explanations/explanations/explanations.py` | **multiple** | Property `_is_thresholded()` | Code |
| `src/calibrated_explanations/core/interval_regressor.py` | 136, 158 | Method parameter names | Code |
| `tests/plugins/test_builtins_behaviour.py` | 400, 447, 499, 545, 592 | Mock property `is_thresholded()` | Test |
| `tests/unit/test_explanations_collection.py` | 262 | Test assertion | Test |

**Total: ~40 references**

### 7.3 Comparative Usage by Document Category

| Category | "Probabilistic Regression" | "Thresholded Regression" | Mixed / Both |
|----------|---------------------------|--------------------------|------------|
| **User Documentation** | ✅ Dominant (~20 refs) | ❌ Rare (1-2 refs) | 0 |
| **ADRs & Architecture** | ⚠️ Some (3-5 refs) | ✅ Dominant (~25 refs) | ~2 |
| **Code (Python)** | ✅ Some (8-10 refs) | ✅ Some (10-12 refs) | N/A |
| **Tests** | ✅ Dominant (~7-10 refs) | ⚠️ Some (2-3 refs) | ~1 |
| **Research Docs** | ⚠️ Some (3-4 refs) | ✅ Dominant (4-5 refs) | 0 |
| **Plugin Analysis** | ⚠️ Some (2-3 refs) | ✅ Some (3-4 refs) | **2 (explicit)** |

---

## Section 8: Transition Plan if "Probabilistic Regression" is Adopted

### Phase 1: Documentation (v0.9.1 pre-release)

**Files to update with terminology clarification:**

1. **ADR-021** – Add terminology section (see Section 6 draft)
2. **ADR-013** – Add cross-reference to terminology
3. **Contributor guide** – Document the naming convention

### Phase 2: Code Naming (v0.9.1)

**Renames:**
- `_is_thresholded()` → `_is_probabilistic_regression()`
- Update docstrings to use "probabilistic regression" as primary term

**Deprecations (optional):**
- Keep `is_thresholded()` as deprecated alias for one release cycle

### Phase 3: Tests (v0.9.1)

**Docstring updates:**
- `test_lime_plugin_probabilistic_regression()` docstring: Change from "Test thresholded regression task" to "Test probabilistic regression (threshold-based probability predictions)"

**New test:**
- Add docstring to at least one test explaining the relationship:
  ```python
  def test_probabilistic_regression_int_threshold_ce(regression_dataset):
      """
      Test probabilistic regression with integer threshold.

      Probabilistic regression (also called thresholded regression in the
      architecture layer) applies a threshold to convert regression predictions
      into calibrated probability predictions. This test validates that
      integer thresholds are accepted.
      """
  ```

### Phase 4: Changelog (v0.9.1 release notes)

```markdown
## Terminology Standardization

- **Breaking change (naming only):** Standardized terminology across documentation
  and code. "Probabilistic regression" is now the canonical user-facing term;
  "thresholded regression" is used in technical architecture documents.
  - `_is_thresholded()` renamed to `_is_probabilistic_regression()`
  - `is_thresholded` parameter/property renamed where user-facing
  - All docstrings updated to prefer "probabilistic regression"
  - See ADR-021 and migration guide for details.

- No API changes; internal renaming only.
```

### Phase 5: Migration Guide

**Create:** `docs/migration/v0.9-to-v0.10-terminology.md`

```markdown
# v0.8 → v0.9 Terminology Changes

## Summary

Terminology was standardized around "probabilistic regression" for user-facing
documentation and "thresholded regression" for technical architecture docs.

## What Changed

### For End Users
- No changes. The `threshold` parameter and `predict_proba(threshold=...)` API remain identical.
- Documentation now consistently uses "probabilistic regression."

### For Contributors & Plugin Developers
- Method `_is_thresholded()` renamed to `_is_probabilistic_regression()`
  - Old name kept as deprecated alias (will be removed in v0.10)
  - Update any custom plugins/code to use new name
- Parameter names (`threshold`, `y_threshold`) unchanged
- ADR-021 clarifies technical terminology

## No Deprecation Policy

This is a documentation and code naming cleanup with no runtime behavior changes.
The deprecation alias for `_is_thresholded()` will be removed in v0.10.
```

---

## Section 9: Conclusion & Action Items

### Findings Summary

1. **"Thresholded regression"** and **"probabilistic regression"** are **synonymous terms** for the same feature
2. **No formal definition** of their equivalence currently exists in the codebase
3. **User-facing docs** consistently use **"probabilistic regression"** (preferred)
4. **Architecture docs (ADRs)** consistently use **"thresholded regression"** (technical)
5. **Implementation code** uses both terms inconsistently, creating potential confusion

### Recommended Action: **Standardize on "Probabilistic Regression"**

**Justification:**
- Alignment with user documentation and marketing
- Consistency with published research
- Clarity for practitioners
- Established precedent in the codebase

### Implementation Checklist

- [ ] **ADR-021:** Add "Terminology" section clarifying the relationship
- [ ] **ADR-013:** Add cross-reference
- [ ] **Code:** Rename `_is_thresholded()` → `_is_probabilistic_regression()`
- [ ] **Docstrings:** Update to prefer "probabilistic regression"
- [ ] **Tests:** Update test docstrings for consistency
- [ ] **CHANGELOG:** Document terminology standardization
- [ ] **Migration guide:** Create `docs/migration/v0.9-to-v0.10-terminology.md`

---

## Appendix A: Search Commands Used

```bash
# All references to either term
grep -r "thresholded.*regression\|probabilistic.*regression" \
  --include="*.md" --include="*.py" --include="*.ipynb"

# Just documentation
grep -r "thresholded.*regression\|probabilistic.*regression" \
  --include="*.md" docs/ improvement_docs/ external_plugins/

# Just code
grep -r "thresholded.*regression\|probabilistic.*regression" \
  --include="*.py" src/ tests/
```

---

## Appendix B: Files Requiring Updates Under Recommended Action

### Must-Update Files:
1. `improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md` – Add terminology section
2. `improvement_docs/adrs/ADR-013-interval-calibrator-plugin-strategy.md` – Add cross-reference
3. `src/calibrated_explanations/explanations/explanations.py` – Rename `_is_thresholded()`
4. `CHANGELOG.md` – Document terminology standardization

### Should-Update Files (docstring clarity):
1. `src/calibrated_explanations/core/interval_regressor.py` – Method docstrings
2. `src/calibrated_explanations/core/calibrated_explainer.py` – Method docstrings
3. `tests/integration/core/test_regression.py` – Test docstrings

### Can-Deprecate (optional):
1. `src/calibrated_explanations/explanations/explanations.py` – Keep `is_thresholded()` as deprecated alias

### Create New:
1. `docs/migration/v0.9-to-v0.10-terminology.md` – Migration guide

---

## Section 10: Implementation Status

**Status:** ✅ **COMPLETE** (November 9, 2025)

### Completed Tasks

- ✅ **ADR-021 Terminology Section Added** (improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md)
  - Added formal "Terminology: Probabilistic Regression vs. Thresholded Regression" section
  - Defines equivalence and clarifies audience/context for each term
  - Provides implementation guidance for contributors

- ✅ **ADR-013 Cross-Reference Added** (improvement_docs/adrs/ADR-013-interval-calibrator-plugin-strategy.md)
  - Added explicit note referencing ADR-021 terminology section
  - Links to official terminology guidance

- ✅ **Code Refactoring: Method Rename** (src/calibrated_explanations/explanations/explanations.py)
  - Renamed `_is_thresholded()` → `_is_probabilistic_regression()` (primary method)
  - Added backward-compatible alias `_is_thresholded()` → delegates to `_is_probabilistic_regression()`
  - Alias will be removed in v0.10.0 when v0.9.0 reaches end-of-life
  - Updated 6 usages to call the new name directly:
    - tests/unit/test_explanations_collection.py (1)
    - tests/integration/core/test_framework.py (4)
    - src/calibrated_explanations/legacy/plotting.py (1, commented)

- ✅ **Docstring Updates** (src/calibrated_explanations/core/)
  - `interval_regressor.py`: Updated `predict_probability()` docstring with probabilistic regression explanation
  - `calibrated_explainer.py`: Updated `_predict()` and `predict()` docstrings; refined parameter descriptions
  - Updated inline comments to use "probabilistic regression" terminology

- ✅ **Test Docstring Clarity** (tests/integration/core/test_regression.py)
  - Enhanced test docstrings in `test_probabilistic_regression_ce()` and `test_probabilistic_regression_int_threshold_ce()`
  - Added explanations linking both terminology terms

- ✅ **Migration Guide Created** (docs/migration/v0.9-to-v0.10-terminology.md)
  - Comprehensive guide explaining changes for different audiences (users, contributors, plugin developers, maintainers)
  - Migration path and timeline for deprecation
  - Rationale for the terminology choice
  - **Indexed and discoverable:**
    - Added to contributor hub (docs/contributor/index.md)
    - Added to main documentation index (docs/index.md) under "Upgrade guides" section
    - Created migration index (docs/migration/index.md) for future migration guides

- ✅ **CHANGELOG Updated** (CHANGELOG.md)
  - Added "Terminology Standardization" section to Unreleased
  - Documented all changes and references

### Verification Checklist

- ✅ Internal method `_is_thresholded()` successfully renamed to `_is_probabilistic_regression()`
- ✅ Backward compatibility alias `_is_thresholded()` added (delegates to new method)
- ✅ All usages of `_is_thresholded()` in code updated to call new method directly
- ✅ Public method `is_thresholded()` on Explanation class **preserved** and unchanged
- ✅ `threshold` and `y_threshold` parameter names **preserved** (describe values, not modes)
- ✅ **Zero breaking changes** to public API
- ✅ Backward compatibility maintained for private methods (via delegation)
- ✅ No behavior changes; terminology-only cleanup
- ✅ ADRs updated with explicit guidance
- ✅ Documentation and tests clarified
- ✅ Migration guide provides clear upgrade path

### Files Modified

| File | Changes |
|------|---------|
| `improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md` | Added terminology section (lines 119-159) |
| `improvement_docs/adrs/ADR-013-interval-calibrator-plugin-strategy.md` | Added cross-reference note |
| `src/calibrated_explanations/explanations/explanations.py` | Added `_is_probabilistic_regression()` method and backward-compatible `_is_thresholded()` alias (lines 446-459) |
| `src/calibrated_explanations/core/interval_regressor.py` | Updated `predict_probability()` docstring |
| `src/calibrated_explanations/core/calibrated_explainer.py` | Updated docstrings and comments (lines 1980-1995, 3243, 3725-3730) |
| `tests/unit/test_explanations_collection.py` | Updated method call (line 262) |
| `tests/integration/core/test_framework.py` | Updated method calls (lines 111, 132, 154, 163) |
| `tests/integration/core/test_regression.py` | Enhanced test docstrings |
| `src/calibrated_explanations/legacy/plotting.py` | Updated comment (line 348) |
| `docs/migration/v0.9-to-v0.10-terminology.md` | **New file** - migration guide |
| `docs/migration/index.md` | **New file** - migration guides index |
| `docs/contributor/index.md` | Added migration reference to toctree |
| `docs/index.md` | Added migration guides section to toctree |
| `CHANGELOG.md` | Added "Terminology Standardization" section |

### Impact Assessment

- **Public API Breaking:** None ✅
- **Private API Breaking:** No (backward-compatible alias provided) ✅
- **Behavior Changes:** None ✅
- **Terminology Changes:** Internal method naming only
  - Affects: Contributors and test code accessing `_is_thresholded()` (private method)
  - Mitigation: Backward-compatible alias provided; existing code continues to work
  - Deprecation: Alias will be removed in v0.10.0

- **User Impact:** Minimal (documentation clarity improvement)
- **Backward Compatibility:** ✅ 100% compatible with v0.9.0

### Next Steps (Future Releases)

- **v0.10.0:** Consider full deprecation lifecycle for internal renaming if future changes warrant it
- **v1.0.0:** Public API fully stabilized; maintain current backward-compatible behavior
- **Future:** If plugin ecosystem grows, expand terminology guidance in plugin development docs

---

**Analysis completed:** November 9, 2025
**Implementation completed:** November 9, 2025
