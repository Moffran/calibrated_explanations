# Installation

Calibrated Explanations is published on PyPI and conda-forge. Install the base
package first, then add extras that unlock plotting, notebook examples, or
contributor tooling.

## PyPI

```bash
pip install calibrated-explanations
```

Extras are opt-in so you only pull the dependencies you need:

| Extra | Purpose | Install command |
| ----- | ------- | --------------- |
| `viz` | Matplotlib-based plotting and PlotSpec adapters. | `pip install "calibrated-explanations[viz]"` |
| `notebooks` | Jupyter notebook tutorials with pinned dependencies. | `pip install "calibrated-explanations[notebooks]"` |
| `dev` | Full development toolchain (linters, docs, tests). | `pip install "calibrated-explanations[dev]"` |
| `eval` | Benchmarking and evaluation tools. | `pip install "calibrated-explanations[eval]"` |
| `external-plugins` | Curated optional bundles (e.g., FAST explanations and intervals). | `pip install "calibrated-explanations[external-plugins]"` |

## conda-forge

```bash
conda install -c conda-forge calibrated-explanations
```

If you rely on extras from PyPI inside a conda environment, install the base
package via conda and then add the relevant extras with `pip`.

## Verifying your environment

```bash
python -c "import calibrated_explanations; print(calibrated_explanations.__version__)"
```

The command should echo `0.10.2` or later once your environment is ready.
