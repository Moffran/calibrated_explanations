"""Shared helpers for the real reject evaluation suite.

The helpers in this module keep all scenario scripts aligned around the same
CE-first workflow:

1. load datasets from an explicit registry,
2. build deterministic proper/calibration/test splits,
3. fit and calibrate a :class:`WrapCalibratedExplainer`,
4. initialize or invoke reject policies through the supported public API,
5. write compact CSV/JSON/Markdown/PNG artifacts for analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunConfig:
    """Execution configuration shared by all reject scenarios."""

    seed: int = 42
    quick: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one evaluation dataset."""

    name: str
    task_type: str
    source: str
    path: Path | None = None
    target: int | str = -1
    drop_columns: tuple[int | str, ...] = ()
    quick: bool = False


@dataclass(frozen=True)
class ClassificationBundle:
    """Prepared binary or multiclass classification evaluation state."""

    dataset_name: str
    feature_names: tuple[str, ...]
    wrapper: Any
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    baseline_pred: np.ndarray
    baseline_proba: np.ndarray


@dataclass(frozen=True)
class RegressionBundle:
    """Prepared regression evaluation state."""

    dataset_name: str
    feature_names: tuple[str, ...]
    wrapper: Any
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    baseline_pred: np.ndarray
    baseline_low: np.ndarray
    baseline_high: np.ndarray
    target_scale: float


def _json_ready(value: Any) -> Any:
    """Convert numpy and pathlib values into JSON-serializable equivalents."""
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _format_scalar(value: Any) -> str:
    """Return a compact markdown-friendly scalar representation."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table_from_df(table: pd.DataFrame, max_rows: int = 12) -> str:
    """Render a small markdown table without external formatting dependencies."""
    if table.empty:
        return "_No rows generated._"
    clipped = table.head(max_rows).copy()
    headers = list(clipped.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in clipped.iterrows():
        lines.append(
            "| " + " | ".join(_format_scalar(row[col]) for col in headers) + " |"
        )
    if len(table) > max_rows:
        lines.append("")
        lines.append(f"_Showing first {max_rows} of {len(table)} rows._")
    return "\n".join(lines)


def save_plot(prefix: str, fig: plt.Figure, suffix: str) -> str:
    """Persist a matplotlib figure under the artifacts directory."""
    path = ARTIFACTS_DIR / f"{prefix}_{suffix}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path.name


def write_csv_json_md(prefix: str, table: pd.DataFrame, meta: Mapping[str, Any]) -> None:
    """Write a scenario artifact bundle."""
    csv_path = ARTIFACTS_DIR / f"{prefix}.csv"
    json_path = ARTIFACTS_DIR / f"{prefix}.json"
    md_path = ARTIFACTS_DIR / f"{prefix}.md"
    serializable_meta = _json_ready(dict(meta))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(serializable_meta, indent=2), encoding="utf-8")

    lines = [f"# {serializable_meta.get('display_name', prefix)}", ""]
    lines.append(f"Rows: {len(table)}")
    lines.append("")

    highlights = serializable_meta.get("highlights") or []
    if highlights:
        lines.extend(["## Key findings", ""])
        lines.extend([f"- {highlight}" for highlight in highlights])
        lines.append("")

    outcome = serializable_meta.get("outcome") or {}
    if outcome:
        lines.extend(["## Outcome snapshot", ""])
        for key, value in outcome.items():
            lines.append(f"- **{key}**: {_format_scalar(value)}")
        lines.append("")

    plots = serializable_meta.get("plots") or []
    if plots:
        lines.extend(["## Plots", ""])
        for plot in plots:
            lines.append(f"- ![{plot}]({plot})")
        lines.append("")

    lines.extend(["## Result table", "", _markdown_table_from_df(table), ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec("breast_cancer", "binary", "sklearn", quick=True),
        DatasetSpec("colic", "binary", "csv", DATA_DIR / "colic.csv", "Y", quick=True),
        DatasetSpec("creditA", "binary", "csv", DATA_DIR / "creditA.csv", "Y"),
        DatasetSpec("diabetes", "binary", "csv", DATA_DIR / "diabetes.csv", "Y", quick=True),
        DatasetSpec("german", "binary", "csv", DATA_DIR / "german.csv", "Y"),
        DatasetSpec("haberman", "binary", "csv", DATA_DIR / "haberman.csv", "Y"),
        DatasetSpec("heartC", "binary", "csv", DATA_DIR / "heartC.csv", "Y"),
        DatasetSpec("heartH", "binary", "csv", DATA_DIR / "heartH.csv", "Y"),
        DatasetSpec("heartS", "binary", "csv", DATA_DIR / "heartS.csv", "Y"),
        DatasetSpec("hepati", "binary", "csv", DATA_DIR / "hepati.csv", "Y"),
        DatasetSpec("iono", "binary", "csv", DATA_DIR / "iono.csv", "Y"),
        DatasetSpec("je4042", "binary", "csv", DATA_DIR / "je4042.csv", "Y"),
        DatasetSpec("je4243", "binary", "csv", DATA_DIR / "je4243.csv", "Y"),
        DatasetSpec("kc1", "binary", "csv", DATA_DIR / "kc1.csv", "Y"),
        DatasetSpec("kc2", "binary", "csv", DATA_DIR / "kc2.csv", "Y"),
        DatasetSpec("kc3", "binary", "csv", DATA_DIR / "kc3.csv", "Y"),
        DatasetSpec("liver", "binary", "csv", DATA_DIR / "liver.csv", "Y"),
        DatasetSpec("pc1req", "binary", "csv", DATA_DIR / "pc1req.csv", "Y"),
        DatasetSpec("pc4", "binary", "csv", DATA_DIR / "pc4.csv", "Y"),
        DatasetSpec("sonar", "binary", "csv", DATA_DIR / "sonar.csv", "Y"),
        DatasetSpec("spect", "binary", "csv", DATA_DIR / "spect.csv", "Y"),
        DatasetSpec("spectf", "binary", "csv", DATA_DIR / "spectf.csv", "Y"),
        DatasetSpec("transfusion", "binary", "csv", DATA_DIR / "transfusion.csv", "Y"),
        DatasetSpec("ttt", "binary", "csv", DATA_DIR / "ttt.csv", "Y"),
        DatasetSpec("vote", "binary", "csv", DATA_DIR / "vote.csv", "Y"),
        DatasetSpec("wbc", "binary", "csv", DATA_DIR / "wbc.csv", "Y"),
        DatasetSpec("balance", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "balance.csv", "Y", quick=True),
        DatasetSpec("cars", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "cars.csv", "Y"),
        DatasetSpec("cmc", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "cmc.csv", "Y"),
        DatasetSpec("cool", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "cool.csv", "Y"),
        DatasetSpec("ecoli", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "ecoli.csv", "Y"),
        DatasetSpec("glass", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "glass.csv", "Y"),
        DatasetSpec("heat", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "heat.csv", "Y"),
        DatasetSpec("image", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "image.csv", "Y"),
        DatasetSpec("iris", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "iris.csv", "Y", quick=True),
        DatasetSpec("steel", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "steel.csv", "Y"),
        DatasetSpec("tae", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "tae.csv", "Y"),
        DatasetSpec("user", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "user.csv", "Y"),
        DatasetSpec("vehicle", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "vehicle.csv", "Y"),
        DatasetSpec("vowel", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "vowel.csv", "Y"),
        DatasetSpec("wave", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "wave.csv", "Y"),
        DatasetSpec("whole", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "whole.csv", "Y"),
        DatasetSpec("wine", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "wine.csv", "Y"),
        DatasetSpec("wineR", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "wineR.csv", "Y"),
        DatasetSpec("wineW", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "wineW.csv", "Y"),
        DatasetSpec("yeast", "multiclass", "csv", DATA_DIR / "Multiclass" / "multi" / "yeast.csv", "Y"),
        DatasetSpec("diabetes_reg", "regression", "sklearn", quick=True),
        DatasetSpec("abalone", "regression", "csv", DATA_DIR / "reg" / "abalone.txt", "REGRESSION", quick=True),
        DatasetSpec("bank8fm", "regression", "csv", DATA_DIR / "reg" / "bank8fm.txt", "REGRESSION"),
        DatasetSpec("bank8nm", "regression", "csv", DATA_DIR / "reg" / "bank8nm.txt", "REGRESSION"),
        DatasetSpec("boston", "regression", "csv", DATA_DIR / "reg" / "boston.txt", "REGRESSION"),
        DatasetSpec("communities", "regression", "csv", DATA_DIR / "reg" / "communities.csv", "ViolentCrimesPerPop", drop_columns=("state", "county", "community", "communityname", "fold")),
        DatasetSpec("comp", "regression", "csv", DATA_DIR / "reg" / "comp.txt", "REGRESSION"),
        DatasetSpec("concreate", "regression", "csv", DATA_DIR / "reg" / "concreate.txt", "REGRESSION"),
        DatasetSpec("cooling", "regression", "csv", DATA_DIR / "reg" / "cooling.txt", "REGRESSION"),
        DatasetSpec("friedm", "regression", "csv", DATA_DIR / "reg" / "friedm.txt", "REGRESSION"),
        DatasetSpec("heating", "regression", "csv", DATA_DIR / "reg" / "heating.txt", "REGRESSION"),
        DatasetSpec("housing", "regression", "csv", DATA_DIR / "reg" / "housing.csv", -1),
        DatasetSpec("kin8fm", "regression", "csv", DATA_DIR / "reg" / "kin8fm.txt", "REGRESSION"),
        DatasetSpec("kin8nm", "regression", "csv", DATA_DIR / "reg" / "kin8nm.txt", "REGRESSION"),
        DatasetSpec("mg", "regression", "csv", DATA_DIR / "reg" / "mg.txt", "REGRESSION"),
        DatasetSpec("plastic", "regression", "csv", DATA_DIR / "reg" / "plastic.txt", "REGRESSION"),
        DatasetSpec("quakes", "regression", "csv", DATA_DIR / "reg" / "quakes.txt", "REGRESSION"),
        DatasetSpec("stock", "regression", "csv", DATA_DIR / "reg" / "stock.txt", "REGRESSION"),
        DatasetSpec("treasury", "regression", "csv", DATA_DIR / "reg" / "treasury.txt", "REGRESSION"),
        DatasetSpec("wineRed", "regression", "csv", DATA_DIR / "reg" / "wineRed.txt", "REGRESSION"),
        DatasetSpec("wineWhite", "regression", "csv", DATA_DIR / "reg" / "wineWhite.txt", "REGRESSION"),
        DatasetSpec("wizmir", "regression", "csv", DATA_DIR / "reg" / "wizmir.txt", "REGRESSION"),
    ]


DATASET_REGISTRY = {spec.name: spec for spec in _dataset_specs()}


def task_specs(task_type: str, *, quick: bool = False) -> list[DatasetSpec]:
    """Return explicit dataset specs for a task type."""
    specs = [spec for spec in DATASET_REGISTRY.values() if spec.task_type == task_type]
    if not quick:
        return specs
    return [spec for spec in specs if spec.quick] or specs[:2]


def _load_builtin_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if spec.name == "breast_cancer":
        ds = load_breast_cancer()
        return ds.data.astype(float), ds.target.astype(int), tuple(ds.feature_names)
    if spec.name == "diabetes_reg":
        x, y = load_diabetes(return_X_y=True)
        features = tuple(f"x{i}" for i in range(x.shape[1]))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), features
    if spec.name == "iris":
        ds = load_iris()
        return ds.data.astype(float), ds.target.astype(int), tuple(ds.feature_names)
    raise KeyError(f"Unsupported builtin dataset {spec.name!r}")


def _read_tabular_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    if frame.shape[1] == 1:
        frame = pd.read_csv(path, sep=";", engine="python")
    return frame


def _resolve_target(frame: pd.DataFrame, target: int | str) -> pd.Series:
    if isinstance(target, str):
        return frame[target]
    return frame.iloc[:, target]


def _drop_columns(frame: pd.DataFrame, drop_columns: Sequence[int | str]) -> pd.DataFrame:
    if not drop_columns:
        return frame
    to_drop: list[str] = []
    for item in drop_columns:
        if isinstance(item, str):
            if item in frame.columns:
                to_drop.append(item)
        else:
            to_drop.append(frame.columns[item])
    return frame.drop(columns=to_drop, errors="ignore")


def _prepare_features(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    return numeric.to_numpy(dtype=float), tuple(str(col) for col in numeric.columns)


def _prepare_target(target: pd.Series, *, regression: bool) -> np.ndarray:
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any():
        numeric = numeric.fillna(numeric.median())
    arr = numeric.to_numpy()
    if regression:
        return arr.astype(float)
    codes, _ = pd.factorize(arr)
    return codes.astype(int)


def load_dataset(spec: DatasetSpec) -> tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load one dataset from the explicit registry."""
    if spec.source == "sklearn":
        x, y, feature_names = _load_builtin_dataset(spec)
        return spec.name, x, y, feature_names

    assert spec.path is not None
    frame = _read_tabular_file(spec.path)
    target = _resolve_target(frame, spec.target)
    features = frame.drop(columns=[spec.target], errors="ignore") if isinstance(spec.target, str) else frame.drop(frame.columns[spec.target], axis=1)
    features = _drop_columns(features, spec.drop_columns)
    x, feature_names = _prepare_features(features)
    y = _prepare_target(target, regression=spec.task_type == "regression")
    return spec.name, x, y, feature_names


def load_binary_datasets(*, quick: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return all configured binary datasets."""
    return [(name, x, y) for name, x, y, _ in (load_dataset(spec) for spec in task_specs("binary", quick=quick))]


def load_multiclass_datasets(*, quick: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return all configured multiclass datasets."""
    return [(name, x, y) for name, x, y, _ in (load_dataset(spec) for spec in task_specs("multiclass", quick=quick))]


def load_regression_datasets(*, quick: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return all configured regression datasets."""
    return [(name, x, y) for name, x, y, _ in (load_dataset(spec) for spec in task_specs("regression", quick=quick))]


def split_dataset(
    x_all: np.ndarray,
    y_all: np.ndarray,
    *,
    seed: int,
    stratify: bool,
    n_cal: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic 60/20/20 splits."""
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.20,
        random_state=seed,
        stratify=y_all if stratify else None,
    )
    cal_fraction = 0.25
    if n_cal is None:
        cal_size: float | int = cal_fraction
    else:
        cal_size = min(max(1, int(n_cal)), len(x_train) - 1)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=cal_size,
        random_state=seed + 1,
        stratify=y_train if stratify else None,
    )
    return x_fit, x_cal, x_test, y_fit, y_cal, y_test


def build_classification_bundle(
    spec: DatasetSpec,
    config: RunConfig,
    *,
    seed_offset: int = 0,
    n_cal: int | None = None,
) -> ClassificationBundle:
    """Train and calibrate a classifier for one registered dataset."""
    dataset_name, x_all, y_all, feature_names = load_dataset(spec)
    seed = config.seed + seed_offset
    x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
        x_all,
        y_all,
        seed=seed,
        stratify=True,
        n_cal=n_cal,
    )
    model = RandomForestClassifier(
        n_estimators=60 if config.quick else 120,
        random_state=seed,
        max_depth=8 if config.quick else None,
        n_jobs=1,
    )
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_fit, y_fit, x_cal, y_cal)
    baseline_pred = model.predict(x_test)
    baseline_proba = model.predict_proba(x_test)
    return ClassificationBundle(
        dataset_name=dataset_name,
        feature_names=feature_names,
        wrapper=wrapper,
        x_fit=x_fit,
        y_fit=y_fit,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        y_test=y_test,
        baseline_pred=np.asarray(baseline_pred),
        baseline_proba=np.asarray(baseline_proba),
    )


def build_regression_bundle(
    spec: DatasetSpec | RunConfig,
    config: RunConfig | None = None,
    *,
    seed_offset: int = 0,
    n_cal: int | None = None,
    learner: Any | None = None,
    explainer_kwargs: Mapping[str, Any] | None = None,
) -> RegressionBundle:
    """Train and calibrate a regressor for one registered dataset."""
    if isinstance(spec, RunConfig):
        config = spec
        spec = task_specs("regression", quick=True)[0]
    assert config is not None
    dataset_name, x_all, y_all_raw, feature_names = load_dataset(spec)
    seed = config.seed + seed_offset
    x_fit, x_cal, x_test, y_fit_raw, y_cal_raw, y_test_raw = split_dataset(
        x_all,
        y_all_raw,
        seed=seed,
        stratify=False,
        n_cal=n_cal,
    )

    y_min = float(np.min(y_fit_raw))
    y_max = float(np.max(y_fit_raw))
    scale = y_max - y_min
    if scale <= 0.0:
        scale = 1.0

    def _norm(arr: np.ndarray) -> np.ndarray:
        return ((np.asarray(arr, dtype=float) - y_min) / scale).astype(float)

    y_fit = _norm(y_fit_raw)
    y_cal = _norm(y_cal_raw)
    y_test = _norm(y_test_raw)
    base_learner = learner
    if base_learner is None:
        base_learner = RandomForestRegressor(
            n_estimators=60 if config.quick else 120,
            random_state=seed,
            max_depth=8 if config.quick else None,
            n_jobs=1,
        )
    model = clone(base_learner)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(
        wrapper,
        x_fit,
        y_fit,
        x_cal,
        y_cal,
        explainer_kwargs={"mode": "regression", **dict(explainer_kwargs or {})},
    )
    baseline_pred, (baseline_low, baseline_high) = wrapper.predict(x_test, uq_interval=True)
    return RegressionBundle(
        dataset_name=dataset_name,
        feature_names=feature_names,
        wrapper=wrapper,
        x_fit=x_fit,
        y_fit=y_fit,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        y_test=y_test,
        baseline_pred=np.asarray(baseline_pred, dtype=float),
        baseline_low=np.asarray(baseline_low, dtype=float),
        baseline_high=np.asarray(baseline_high, dtype=float),
        target_scale=float(scale),
    )


def build_binary_bundle(config: RunConfig, *, seed_offset: int = 0) -> ClassificationBundle:
    """Train and calibrate a binary classifier on the quick binary reference set."""
    return build_classification_bundle(task_specs("binary", quick=True)[0], config, seed_offset=seed_offset)


def build_multiclass_bundle(config: RunConfig, *, seed_offset: int = 0) -> ClassificationBundle:
    """Train and calibrate a multiclass classifier on the quick multiclass reference set."""
    specs = task_specs("multiclass", quick=True)
    chosen = next((spec for spec in specs if spec.name == "iris"), specs[0])
    return build_classification_bundle(chosen, config, seed_offset=seed_offset)


def build_regression_bundle_default(config: RunConfig, *, seed_offset: int = 0) -> RegressionBundle:
    """Train and calibrate a regression explainer on the quick regression reference set."""
    return build_regression_bundle(task_specs("regression", quick=True)[0], config, seed_offset=seed_offset)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute a simple expected calibration error for binary probabilities."""
    if len(y_true) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if not np.any(in_bin):
            continue
        avg_conf = float(np.mean(y_prob[in_bin]))
        avg_acc = float(np.mean(y_true[in_bin]))
        error += abs(avg_acc - avg_conf) * float(np.mean(in_bin))
    return float(error)


def explanation_count(explanation: Any) -> int:
    """Return the number of explanations carried by a collection-like object."""
    if explanation is None:
        return 0
    if hasattr(explanation, "explanations"):
        return len(explanation.explanations)
    try:
        return len(explanation)
    except TypeError:
        return 1


def confidence_from_matrix(proba: np.ndarray) -> np.ndarray:
    """Return per-instance max probability/confidence."""
    arr = np.asarray(proba)
    if arr.ndim == 1:
        return np.maximum(arr, 1.0 - arr)
    return np.max(arr, axis=1)


def interval_outside_fraction(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Return the fraction of points outside a prediction interval."""
    outside = (y_true < low) | (y_true > high)
    return float(np.mean(outside))


def interval_width_mean(low: np.ndarray, high: np.ndarray) -> float:
    """Return mean interval width."""
    return float(np.mean(np.asarray(high) - np.asarray(low)))


def reject_breakdown(
    wrapper: Any,
    inputs: np.ndarray,
    *,
    confidence: float,
    threshold: float | None = None,
    ncf: str | None = None,
    w: float = 0.5,
) -> dict[str, Any]:
    """Return the detailed reject breakdown through the supported orchestrator."""
    assert wrapper.explainer is not None
    if getattr(wrapper.explainer, "reject_learner", None) is None:
        wrapper.explainer.reject_orchestrator.initialize_reject_learner(
            threshold=threshold, ncf=ncf, w=w
        )
    return wrapper.explainer.reject_orchestrator.predict_reject_breakdown(
        inputs,
        confidence=confidence,
        threshold=threshold,
    )


def select_best_row(
    table: pd.DataFrame,
    *,
    sort_by: Sequence[str],
    ascending: Sequence[bool],
) -> pd.Series:
    """Return the best row under a simple lexicographic ranking."""
    ranked = table.sort_values(list(sort_by), ascending=list(ascending), kind="mergesort")
    return ranked.iloc[0]


def quantile_grid(quick: bool, *, start: float = 0.10, stop: float = 0.50) -> np.ndarray:
    """Return a deterministic threshold-quantile grid."""
    count = 3 if quick else 5
    return np.linspace(start, stop, count)


def confidence_grid(quick: bool, *, start: float = 0.80, stop: float = 0.99) -> np.ndarray:
    """Return a deterministic confidence sweep grid."""
    count = 4 if quick else 9
    return np.linspace(start, stop, count)


def seed_grid(config: RunConfig, *, count_quick: int = 2, count_full: int = 5) -> Iterable[int]:
    """Return the deterministic seed offsets used in repeated experiments."""
    count = count_quick if config.quick else count_full
    return range(count)


def binary_accuracy_from_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
) -> float:
    """Evaluate regression predictions as thresholded binary decisions."""
    return float(
        accuracy_score((y_true >= threshold).astype(int), (y_pred >= threshold).astype(int))
    )


def regression_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error as float."""
    return float(mean_squared_error(y_true, y_pred))


def regression_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute error as float."""
    return float(mean_absolute_error(y_true, y_pred))


def accepted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, accepted: np.ndarray) -> float:
    """Return accepted-set accuracy."""
    if not np.any(accepted):
        return float("nan")
    return float(np.mean(np.asarray(y_true)[accepted] == np.asarray(y_pred)[accepted]))


def empirical_coverage(prediction_set: np.ndarray, y_true: np.ndarray) -> float:
    """Return empirical label-set coverage."""
    if len(y_true) == 0:
        return float("nan")
    rows = np.arange(len(y_true))
    return float(np.mean(np.asarray(prediction_set, dtype=bool)[rows, np.asarray(y_true, dtype=int)]))


def clopper_pearson_interval(successes: int, total: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Return an exact Clopper-Pearson interval."""
    if total <= 0:
        return float("nan"), float("nan")
    res = binomtest(successes, total)
    interval = res.proportion_ci(confidence_level=confidence, method="exact")
    return float(interval.low), float(interval.high)


def feature_weight_matrix(explanations: Any) -> np.ndarray:
    """Return per-instance explanation weights as a dense matrix."""
    rows: list[np.ndarray] = []
    for explanation in explanations:
        weights = getattr(explanation, "feature_weights", None)
        if not isinstance(weights, dict):
            continue
        predict_weights = weights.get("predict")
        if predict_weights is None:
            continue
        rows.append(np.asarray(predict_weights, dtype=float))
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def mean_feature_weight_variance(explanations: Any) -> float:
    """Return mean variance of feature weights across instances."""
    matrix = feature_weight_matrix(explanations)
    if matrix.size == 0 or matrix.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.var(matrix, axis=0)))


def difficulty_threshold_categorizer(
    difficulty_apply: Callable[[np.ndarray], np.ndarray],
    threshold: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a two-bin Mondrian categorizer from a difficulty threshold."""

    def _categorizer(x: np.ndarray) -> np.ndarray:
        scores = np.asarray(difficulty_apply(x), dtype=float)
        return (scores > threshold).astype(int)

    return _categorizer


def predicted_value_categorizer(
    predictor: Callable[[np.ndarray], np.ndarray],
    edges: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a binning categorizer from predicted values."""

    def _categorizer(x: np.ndarray) -> np.ndarray:
        pred = np.asarray(predictor(x), dtype=float)
        bins = np.digitize(pred, edges[1:-1], right=True)
        return bins.astype(int)

    return _categorizer


def accepted_interval_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, float]:
    """Return accepted-subset interval and error metrics."""
    if not np.any(accepted):
        return {
            "accepted_coverage": float("nan"),
            "accepted_interval_width": float("nan"),
            "accepted_mse": float("nan"),
            "accepted_mae": float("nan"),
        }
    return {
        "accepted_coverage": 1.0
        - interval_outside_fraction(y_true[accepted], low[accepted], high[accepted]),
        "accepted_interval_width": interval_width_mean(low[accepted], high[accepted]),
        "accepted_mse": regression_mse(y_true[accepted], pred[accepted]),
        "accepted_mae": regression_mae(y_true[accepted], pred[accepted]),
    }
