# Theory & literature

Calibrated Explanations builds on a multi-year research program spanning binary
and multiclass classification, probabilistic and interval regression, and
uncertainty-aware interpretation tooling. Use this page to locate published
papers, benchmark references, and funding acknowledgements.

## Publication highlights

### Peer-reviewed work
- **Trustworthy explanations dissertation (2023):** Introduces the
  decision-support principles behind calibrated explanations and documents the
  evaluation scaffolding used for subsequent studies. Accessible via the Jönköping
  University repository.
- **Expert Systems with Applications (2024):** Binary-classification study with
  calibrated factual and alternative rules across 25 benchmarks. DOI:
  [`10.1016/j.eswa.2024.123154`](https://doi.org/10.1016/j.eswa.2024.123154).
- **Machine Learning journal (2025):** Regression and probabilistic regression
  extension with interval guarantees mirroring the classification workflow. DOI:
  [`10.1007/s10994-024-06642-8`](https://doi.org/10.1007/s10994-024-06642-8).
- **xAI 2024 proceedings:** Conditional calibrated explanations for fairness-aware
  deployments. DOI:
  [`10.1007/978-3-031-63787-2_17`](https://doi.org/10.1007/978-3-031-63787-2_17).
- **PMLR 230 (2024):** Multiclass calibration analysis with empirical guarantees
  and tutorial material. Proceedings and artefacts available through the
  [PMLR volume](https://proceedings.mlr.press/v230/lofstrom24a.html) and
  [conference slides](https://copa-conference.com/presentations/Lofstrom.pdf).
- **Annals of Mathematics and Artificial Intelligence (2023):** First study that
  links calibration quality to explanation reliability. DOI:
  [`10.1007/s10472-023-09837-2`](https://doi.org/10.1007/s10472-023-09837-2).

### Preprints and in-flight studies
- **Ensured explanations (2024):** Investigates strategies that actively reduce
  epistemic uncertainty in the generated alternatives. Preprint:
  [`arXiv:2410.05479`](https://arxiv.org/abs/2410.05479).
- **Fast calibrated explanations (2024):** Plugin-oriented acceleration of rule
  generation while preserving uncertainty guarantees. DOI:
  [`10.1007/978-3-032-08333-3_16`](https://doi.org/10.1007/978-3-032-08333-3_16).

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
- Browse the [project README](https://github.com/Moffran/calibrated_explanations/blob/main/README.md)
  for end-to-end walkthroughs that demonstrate how the research artifacts surface
  in day-to-day usage.
- Visit the {doc}`../concepts/probabilistic_regression` guide to see how the
  regression research surfaces in runtime APIs and notebooks.
