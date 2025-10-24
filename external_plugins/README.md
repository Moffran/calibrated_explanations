# External Plugins Placeholder

This directory lists first-party maintained external plugins that adhere to the calibrated explanations contract. Populate the
folder with subdirectories or metadata files (e.g., `plugin-name/README.md`) for each plugin prior to publishing the aggregated
extras installation path.

## Installation extras

All vetted plugins should be installable via the aggregated extras group:

```
pip install calibrated-explanations[external-plugins]
```

Document each plugin's purpose, calibration guarantees, and research references inside its subdirectory. Plugins must preserve
calibrated factual and alternative explanations (with triangular plots for alternatives) and clearly mark telemetry or other
extras as optional.
