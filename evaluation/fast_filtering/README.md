# Fast Filtering Evaluation

This folder contains multi-dataset evaluation scripts for fast feature filtering.

## Quick start
- Run from the repository root so `evaluation` is importable.
- Use module execution to avoid `ModuleNotFoundError: No module named 'evaluation'`.

```bash
python -m evaluation.fast_filtering.setup_ablation_multi
python -m evaluation.fast_filtering.fast_feature_filtering_ablation_multi
python -m evaluation.fast_filtering.fast_filtering_feature_overlap
python -m evaluation.fast_filtering.fast_filtering_topk_sweep
python -m evaluation.fast_filtering.synthetic_feature_sweep
```

If you must run a script by path, set `PYTHONPATH=.` first:

```bash
set PYTHONPATH=.
python evaluation/fast_filtering/setup_ablation_multi.py
```

## Scripts
- `setup_ablation_multi.py`
  - Purpose: compare explainer setups (legacy, sequential, parallel, caching, fast_filtering).
  - Output: `evaluation/fast_filtering/setup_ablation_multi_results.json`.
  - Parameters:

| Parameter | Default | Type / allowed values | Notes |
| --- | --- | --- | --- |
| `--tasks` | `classification multiclass regression` | list[str]; `classification`, `multiclass`, `regression` | Select dataset groups. |
| `--datasets` | `None` | list[str] | Subset of dataset names (see `evaluation/fast_filtering/dataset_utils.py`). |
| `--limit` | `None` | int | Limit number of datasets after filtering. |
| `--models` | `RF HGB` | list[str]; `RF`, `HGB` | RandomForest or HistGradientBoosting. |
| `--workers` | `6` | int | Used for instance-parallel runs. |
| `--instance-chunk-size` | `50` | int | Batch size for instance-parallel runs. |
| `--test-size` | `0.25` | float in (0,1) | Held-out test split. |
| `--calibration-size` | `0.2` | float in (0,1) | Calibration split from training data. |
| `--max-samples` | `None` | int | Cap dataset size after loading. |
| `--fast-filter-top-k` | `8` | int >= 1 | Per-instance top-k for fast filtering. |
| `--enable-instance-telemetry` | `False` | flag | Enables per-instance telemetry (may fail in parallel runs). |
| `--results-file` | `evaluation/fast_filtering/setup_ablation_multi_results.json` | str (path) | Output JSON path. |

- `fast_feature_filtering_ablation_multi.py`
  - Purpose: timing ablation with and without fast filtering, optionally including
    `explain_fast`, `explore_alternatives`, and comparator methods.
  - Output: `evaluation/fast_filtering/fast_filtering_ablation_multi_results.json`.
  - Parameters:

| Parameter | Default | Type / allowed values | Notes |
| --- | --- | --- | --- |
| `--tasks` | `classification multiclass regression` | list[str]; `classification`, `multiclass`, `regression` | Select dataset groups. |
| `--datasets` | `None` | list[str] | Subset of dataset names (see `evaluation/fast_filtering/dataset_utils.py`). |
| `--limit` | `None` | int | Limit number of datasets after filtering. |
| `--models` | `RF HGB` | list[str]; `RF`, `HGB` | RandomForest or HistGradientBoosting. |
| `--test-size` | `0.25` | float in (0,1) | Held-out test split. |
| `--calibration-size` | `0.2` | float in (0,1) | Calibration split from training data. |
| `--max-samples` | `None` | int | Cap dataset size after loading. |
| `--max-explain` | `250` | int | Limit number of test instances to explain. |
| `--fast-filter-top-k` | `8` | int >= 1 | Per-instance top-k for fast filtering. |
| `--include-fast` | `False` | flag | Include `explain_fast` timings. |
| `--include-alternatives` | `False` | flag | Run `explore_alternatives` (classification by default). |
| `--allow-regression-alternatives` | `False` | flag | Allow alternatives for regression tasks. |
| `--include-lime` | `False` | flag | Include LIME comparator (requires `lime`). |
| `--include-shap` | `False` | flag | Include SHAP comparator (requires `shap`). |
| `--results-file` | `evaluation/fast_filtering/fast_filtering_ablation_multi_results.json` | str (path) | Output JSON path. |

- `fast_filtering_feature_overlap.py`
  - Purpose: measure feature-overlap metrics between full and fast-filtered explanations
    (Jaccard, top-k inclusion, rank correlation).
  - Output: `evaluation/fast_filtering/fast_filtering_feature_overlap_results.json`.
  - Parameters:

| Parameter | Default | Type / allowed values | Notes |
| --- | --- | --- | --- |
| `--tasks` | `classification multiclass regression` | list[str]; `classification`, `multiclass`, `regression` | Select dataset groups. |
| `--datasets` | `None` | list[str] | Subset of dataset names (see `evaluation/fast_filtering/dataset_utils.py`). |
| `--limit` | `None` | int | Limit number of datasets after filtering. |
| `--models` | `RF HGB` | list[str]; `RF`, `HGB` | RandomForest or HistGradientBoosting. |
| `--test-size` | `0.25` | float in (0,1) | Held-out test split. |
| `--calibration-size` | `0.2` | float in (0,1) | Calibration split from training data. |
| `--max-samples` | `None` | int | Cap dataset size after loading. |
| `--max-explain` | `250` | int | Limit number of test instances to explain. |
| `--fast-filter-top-k` | `8` | int >= 1 | Per-instance top-k for fast filtering. |
| `--include-alternatives` | `False` | flag | Run `explore_alternatives` (classification by default). |
| `--allow-regression-alternatives` | `False` | flag | Allow alternatives for regression tasks. |
| `--save-instance-metrics` | `False` | flag | Persist per-instance overlap metrics in output. |
| `--results-file` | `evaluation/fast_filtering/fast_filtering_feature_overlap_results.json` | str (path) | Output JSON path. |

- `fast_filtering_topk_sweep.py`
  - Purpose: sweep per-instance top-k to show quality/speed tradeoffs.
  - Output: `evaluation/fast_filtering/fast_filtering_topk_sweep_results.json`.
  - Parameters:

| Parameter | Default | Type / allowed values | Notes |
| --- | --- | --- | --- |
| `--tasks` | `classification multiclass regression` | list[str]; `classification`, `multiclass`, `regression` | Select dataset groups. |
| `--datasets` | `None` | list[str] | Subset of dataset names (see `evaluation/fast_filtering/dataset_utils.py`). |
| `--limit` | `None` | int | Limit number of datasets after filtering. |
| `--models` | `RF HGB` | list[str]; `RF`, `HGB` | RandomForest or HistGradientBoosting. |
| `--test-size` | `0.25` | float in (0,1) | Held-out test split. |
| `--calibration-size` | `0.2` | float in (0,1) | Calibration split from training data. |
| `--max-samples` | `None` | int | Cap dataset size after loading. |
| `--max-explain` | `250` | int | Limit number of test instances to explain. |
| `--top-k-values` | `2 4 8 16` | list[int] >= 1 | Per-instance top-k sweep values. |
| `--include-alternatives` | `False` | flag | Run `explore_alternatives` (classification by default). |
| `--allow-regression-alternatives` | `False` | flag | Allow alternatives for regression tasks. |
| `--results-file` | `evaluation/fast_filtering/fast_filtering_topk_sweep_results.json` | str (path) | Output JSON path. |

- `analyze_results.py`
  - Purpose: generate markdown tables from the JSON results and optionally plot top-k summaries.
  - Output: markdown tables to stdout or to `--out-dir`; optional plots `topk_time_ratio.png`,
    `topk_overlap.png`.
  - Parameters:

| Parameter | Default | Type / allowed values | Notes |
| --- | --- | --- | --- |
| `--ablation` | `evaluation/fast_filtering/fast_filtering_ablation_multi_results.json` | str (path) | Input JSON from ablation. |
| `--overlap` | `evaluation/fast_filtering/fast_filtering_feature_overlap_results.json` | str (path) | Input JSON from overlap. |
| `--topk` | `evaluation/fast_filtering/fast_filtering_topk_sweep_results.json` | str (path) | Input JSON from top-k sweep. |
| `--synthetic` | `evaluation/fast_filtering/synthetic_feature_sweep_results.json` | str (path) | Input JSON from synthetic sweep (optional). |
| `--out-dir` | `None` | str (path) | Write markdown tables to this directory. |
| `--plots` | `False` | flag | Write PNG plots (requires `matplotlib`). |

- `synthetic_feature_sweep.py`
  - Purpose: sweep feature counts on synthetic datasets to study runtime/overlap vs dimensionality.
  - Output: `evaluation/fast_filtering/synthetic_feature_sweep_results.json`.
  - Parameters:

| Parameter | Default | Type / allowed values | Notes |
| --- | --- | --- | --- |
| `--tasks` | `classification regression` | list[str]; `classification`, `regression` | Synthetic task types. |
| `--feature-counts` | `20 50 100 200 500` | list[int] >= 2 | Feature counts to sweep. |
| `--n-samples` | `2000` | int >= 1 | Total synthetic samples. |
| `--n-informative` | `10` | int >= 1 | Informative feature count (capped to n_features-1). |
| `--n-redundant` | `0` | int >= 0 | Redundant feature count (capped). |
| `--n-repeated` | `0` | int >= 0 | Repeated feature count (capped). |
| `--n-classes` | `2` | int >= 2 | Classes for classification. |
| `--noise` | `0.1` | float >= 0 | Label noise / regression noise. |
| `--models` | `RF HGB` | list[str]; `RF`, `HGB` | RandomForest or HistGradientBoosting. |
| `--test-size` | `0.25` | float in (0,1) | Held-out test split. |
| `--calibration-size` | `0.2` | float in (0,1) | Calibration split from training data. |
| `--max-explain` | `250` | int | Limit number of test instances to explain. |
| `--top-k-values` | `4 8 16` | list[int] >= 1 | Per-instance top-k sweep values. |
| `--random-state` | `42` | int | RNG seed. |
| `--results-file` | `evaluation/fast_filtering/synthetic_feature_sweep_results.json` | str (path) | Output JSON path. |

## Interpretation Key
- **with_filtering / without_filtering**: Runtime in seconds for the respective mode.
- **ratio**: `with_filtering / without_filtering`. Values < 1.0 indicate speedup.
- **jaccard_topk**: Jaccard similarity between the sets of top-k features. 1.0 is perfect overlap.
- **topk_inclusion**: Fraction of the "true" top-k features (from the full run) that were preserved in the filtered run.
- **spearman_rank**: Rank correlation of feature importance scores between full and filtered runs.
- **kept_feature_count**: The average number of features that passed the fast-filtering threshold.

## Setup Key
- **legacy**: Uses the original `CalibratedExplainer` implementation (pre-orchestrator).
- **sequential**: Uses the new `ExplanationOrchestrator` in sequential mode.
- **instance_parallel**: Uses the new `ExplanationOrchestrator` with instance-level parallelism.
- **caching**: Uses the new `ExplanationOrchestrator` with rule caching enabled.
- **fast_filtering**: Uses the new `ExplanationOrchestrator` with fast feature filtering enabled.

## Research Contribution Guidance
To frame these results as a strong research contribution, consider the following mapping of experiments to core arguments:

### 1. Performance & Scalability (The "Efficiency" Argument)
- **Experiment**: `setup_ablation_multi.py` and `fast_feature_filtering_ablation_multi.py`.
- **Guidance**: Use these to demonstrate that the proposed architecture significantly reduces the computational overhead of calibrated explanations. Highlight the cumulative speedup when combining parallelism, caching, and fast filtering. Compare against the `legacy` baseline to show the evolution of the framework.

### 2. Fidelity & Approximation Quality (The "Faithfulness" Argument)
- **Experiment**: `fast_filtering_feature_overlap.py`.
- **Guidance**: A common critique of "fast" methods is that they sacrifice accuracy. Use the Jaccard similarity and Top-K inclusion metrics to prove that the fast-filtering path preserves the most important features identified by the full (exhaustive) path. High Spearman rank correlation further supports that the relative importance of features remains consistent.

### 3. Parameter Sensitivity & Pareto Frontier (The "Usability" Argument)
- **Experiment**: `fast_filtering_topk_sweep.py`.
- **Guidance**: Show the "elbow" in the quality-vs-speed tradeoff. This demonstrates that a small `top_k` (e.g., 8 or 16) captures the vast majority of the explanation's value while providing near-maximal speedup. This justifies the default settings and provides a clear guide for practitioners.

### 4. Dimensionality Scaling (The "High-Dimensional" Argument)
- **Experiment**: `synthetic_feature_sweep.py`.
- **Guidance**: This is critical for showing *why* fast filtering is necessary. As the number of features increases, the exhaustive search becomes prohibitively slow. Use this sweep to show that the speedup ratio improves as dimensionality grows, while the overlap metrics remain stable, proving the method's robustness in high-dimensional spaces.

### 5. Generalizability (The "Robustness" Argument)
- **Experiment**: All multi-dataset runs.
- **Guidance**: By reporting results across `classification`, `multiclass`, and `regression` tasks on diverse real-world datasets, you demonstrate that the contribution is not overfitted to a specific domain or task type.

## Datasets
Datasets are loaded from `data/` using `evaluation/fast_filtering/dataset_utils.py`.
If a dataset is missing on disk, the script will fail when it is selected.

To limit the workload:

```bash
python -m evaluation.fast_filtering.fast_feature_filtering_ablation_multi --tasks classification --limit 3
python -m evaluation.fast_filtering.fast_filtering_feature_overlap --datasets german diabetes
python -m evaluation.fast_filtering.fast_filtering_topk_sweep --top-k-values 2 4 8
```

## Optional dependencies
The LIME and SHAP comparators are optional:

```bash
python -m evaluation.fast_filtering.fast_feature_filtering_ablation_multi --include-lime --include-shap
```

Install `lime` and `shap` if you enable them.

## Troubleshooting
- If `instance_parallel` fails with rule payload errors, rerun without
  per-instance telemetry (default). Use `--enable-instance-telemetry` only if
  you need telemetry fields in the results.
