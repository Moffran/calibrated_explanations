# Future Work & Research Directions

This ledger tracks potential research directions and enhancements to the calibrated explanations framework, organized by theme and linked to relevant literature.

## Active Research Areas

### 1. Adaptive Binning Strategies

**Current state:** Fixed or Mondrian-based discretization.

**Research question:** Can we develop adaptive binning that optimizes for both explanation fidelity and computational efficiency?

**Literature connections:**
- Conformal prediction literature on adaptive binning
- Information-theoretic approaches to discretization

**Potential implementation:** v0.10.x or later

---

### 2. Multi-Calibration Fairness

**Current state:** Basic Mondrian categorizer support for conditional fairness.

**Research question:** How can we extend the framework to guarantee multi-calibration across intersectional protected attributes?

**Literature connections:**
- Multi-calibration (Hébert-Johnson et al., 2018)
- Intersectional fairness frameworks

**Potential implementation:** v1.1.x (requires ADR for fairness primitives)

---

### 3. Conformal Guarantees for Explanations

**Current state:** Probabilistic intervals for predictions.

**Research question:** Can we provide distribution-free coverage guarantees for feature importance rankings?

**Literature connections:**
- Conformal prediction theory
- Venn-Abers calibration

**Potential implementation:** Research prototype first

---

### 4. Temporal Calibration Drift

**Current state:** Static calibration data.

**Research question:** How should the framework handle concept drift and when should recalibration be triggered?

**Literature connections:**
- Online conformal prediction
- Adaptive learning systems

**Potential implementation:** v1.2.x (monitoring tooling)

---

### 5. High-Dimensional Feature Interactions

**Current state:** Pairwise conjunctions via `add_conjunctions()`.

**Research question:** What are computationally feasible approaches for explaining higher-order interactions while maintaining calibration guarantees?

**Literature connections:**
- Shapley interaction indices
- Functional ANOVA decompositions

**Potential implementation:** Plugin system extension (v1.x)

---

## Long-Term Vision

### Theoretical Foundations
- Formal analysis of calibration error bounds for explanations
- Convergence guarantees for iterative calibration methods
- Connections to information theory and rate-distortion tradeoffs

### Computational Efficiency
- Parallel calibration strategies (continuation of ADR-004)
- Incremental calibration for streaming data
- GPU acceleration for large-scale deployments

### Ecosystem Integration
- Standardized interchange formats for calibrated explanations
- Integration with MLOps platforms (MLflow, Weights & Biases)
- AutoML-friendly presets and hyperparameter guidance

---

## Contributing Research

We welcome research contributions! If you're working on any of these areas or have related ideas:

1. Open a discussion in the [GitHub Discussions](https://github.com/Moffran/calibrated_explanations/discussions) forum
2. Reference relevant ADRs (see `docs/improvement/adrs/`)
3. Consider prototyping as a plugin (see ADR-006, ADR-013, ADR-015)
4. For published work, cite the calibrated explanations framework and we'll add your paper to our literature tracker

---

## References

This ledger complements the [citing guide](../citing.md) which lists foundational papers and recommended citation practices.
