# Theory & literature

Calibrated Explanations builds on a multi-year research program spanning binary
and multiclass classification, probabilistic and interval regression, and
uncertainty-aware interpretation tooling. Use this page to locate published
papers, benchmark references, and funding acknowledgements.

## Publication highlights

### Peer-reviewed work
- **Trustworthy explanations dissertation (2023):** Introduces the
  decision-support principles behind calibrated explanations and documents the
  evaluation scaffolding used for subsequent studies.
- **Expert Systems with Applications (2024):** Establishes calibrated
  explanations for binary classification with uncertainty-aware factual and
  alternative rules, validated on 25 benchmark datasets and runtime stability
  checks.
- **Machine Learning journal (2025):** Extends the method to regression and
  probabilistic regression with interval guarantees that match the classification
  results.
- **xAI 2024 proceedings:** Demonstrates conditional calibrated explanations for
  fairness-sensitive workflows.
- **PMLR 230 (2024):** Adds multi-class coverage with empirical calibration and
  tutorial material (paper and slides) for researchers adopting the method.
- **Annals of Mathematics and Artificial Intelligence (2023):** Documents the
  original study tying calibration quality to explanation reliability, including
  public code for reproducing the experiments.

### Preprints and in-flight studies
- **Ensured explanations (2024):** Investigates strategies that actively reduce
  epistemic uncertainty in the generated alternatives.
- **Fast calibrated explanations (2024):** Shares a plugin-oriented approach for
  speeding up rule generation while preserving uncertainty guarantees.

## Benchmarks and evaluations

Calibrated Explanations has been stress-tested on 25 publicly available
benchmark datasets for binary classification, with the same evaluation harness
reused for regression and probabilistic extensions. The replication packages
linked from the journal papers include dataset manifests, metric definitions,
plots, and supporting notebooks so you can compare against your own workloads.

## Funding and acknowledgements

The research has been funded by the [Swedish Knowledge
Foundation](https://www.kks.se/) through the Knowledge Intensive Product
Realization SPARK environment at Jönköping University. Grants include AFAIR
(20200223), ETIAI (20230040), PREMACOP (20220187), and the Industrial Graduate
School in Digital Retailing (INSiDR, 20160035). The project also thanks
contributors for engineering support, including release automation, regression
tooling, and third-party libraries such as `crepes`, `venn-abers`,
`ConformaSight`, and scikit-learn components that ship with the distribution.

## Keep exploring

- Dive into the full citation list and BibTeX entries in {doc}`../citing`.
- Browse the [project README](https://github.com/Moffran/calibrated_explanations#readme)
  for end-to-end walkthroughs that demonstrate how the research artifacts surface
  in day-to-day usage.
- Visit the {doc}`../concepts/probabilistic_regression` guide to see how the
  regression research surfaces in runtime APIs and notebooks.
