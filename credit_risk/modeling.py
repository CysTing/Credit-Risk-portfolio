"""Reproducible train/calibration/test experiment, without model pickles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import FINANCIAL_FEATURES, PAY_FEATURES, TARGET, load_dataset
from .features import FinancialFeatures, make_preprocessor

SEED = 42
MODEL_NAMES = ("Logistic regression", "Random forest")


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_cal: pd.Series
    y_test: pd.Series


@dataclass
class ModelBundle:
    models: dict[str, Pipeline]
    calibrated: dict[str, CalibratedClassifierCV]
    baseline: DummyClassifier
    splits: SplitData
    reference: pd.DataFrame
    dataset: pd.DataFrame


def split_data(data: pd.DataFrame, random_state: int = SEED) -> SplitData:
    """Stratify 60% fit / 20% calibration / 20% final test with stable indices."""
    X, y = data[FINANCIAL_FEATURES], data[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=random_state,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=random_state,
    )
    return SplitData(X_train, X_cal, X_test, y_train, y_cal, y_test)


def financial_history_overlap(splits: SplitData) -> dict:
    """Measure identical 19-field histories across the fixed random splits.

    Hashes exclude row indices and outcomes. These are repeated observed
    histories, not evidence that rows refer to the same person. Counts use
    pandas' deterministic 64-bit row hashes and do not disclose individual rows.
    """
    hashes = {
        "train": pd.util.hash_pandas_object(splits.X_train[FINANCIAL_FEATURES], index=False),
        "calibration": pd.util.hash_pandas_object(splits.X_cal[FINANCIAL_FEATURES], index=False),
        "test": pd.util.hash_pandas_object(splits.X_test[FINANCIAL_FEATURES], index=False),
    }
    unique = {name: set(values) for name, values in hashes.items()}
    return {
        "definition": "Identical 19 financial feature values only; target and index excluded; pandas 64-bit row hashes.",
        "unique_histories": {name: len(values) for name, values in unique.items()},
        "unique_train_calibration_intersection": len(unique["train"] & unique["calibration"]),
        "unique_train_test_intersection": len(unique["train"] & unique["test"]),
        "unique_calibration_test_intersection": len(unique["calibration"] & unique["test"]),
        "test_rows_with_history_in_train": int(hashes["test"].isin(unique["train"]).sum()),
        "test_rows_with_history_in_calibration": int(hashes["test"].isin(unique["calibration"]).sum()),
    }


def build_pipeline(model_name: str, n_estimators: int = 120) -> Pipeline:
    """Use fixed hyperparameters chosen before looking at the final test set."""
    if model_name == "Logistic regression":
        estimator = LogisticRegression(C=1., max_iter=1500, solver="lbfgs", random_state=SEED)
    elif model_name == "Random forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=10, min_samples_leaf=15,
            max_features=.7, n_jobs=2, random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([
        ("features", FinancialFeatures()),
        ("preprocess", make_preprocessor()),
        ("classifier", estimator),
    ])


def train_models(data: pd.DataFrame | None = None, n_estimators: int = 120) -> ModelBundle:
    """Fit on training rows and sigmoid-calibrate on disjoint calibration rows.

    Both model families are reported; neither is silently selected by test AUC.
    Class prevalence is left unchanged (no oversampling or class weighting).
    The app caches this return value in memory and never unpickles user files.
    """
    frame = load_dataset() if data is None else data.copy()
    splits = split_data(frame)
    models, calibrated = {}, {}
    for name in MODEL_NAMES:
        raw = build_pipeline(name, n_estimators=n_estimators)
        raw.fit(splits.X_train, splits.y_train)
        cal = CalibratedClassifierCV(FrozenEstimator(raw), method="sigmoid")
        cal.fit(splits.X_cal, splits.y_cal)
        models[name], calibrated[name] = raw, cal
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(splits.X_train, splits.y_train)
    # Synthetic reference summarizes training rows; categorical modes preserve
    # valid status codes even when their arithmetic median would be fractional.
    reference = splits.X_train.median().to_frame().T[FINANCIAL_FEATURES]
    for col in PAY_FEATURES:
        modes = splits.X_train[col].mode()
        reference.loc[reference.index[0], col] = modes.iloc[0] if len(modes) else 0
    return ModelBundle(models, calibrated, baseline, splits, reference, frame)


def metrics_at_threshold(y: pd.Series | np.ndarray, prob: np.ndarray, threshold: float = .5) -> dict:
    """Separate probability quality from threshold-dependent classification."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie between 0 and 1.")
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flag_rate": float(pred.mean()), "n": int(len(y)),
    }


def evaluate_models(bundle: ModelBundle, threshold: float = .5) -> pd.DataFrame:
    """Return five held-out rows: raw/calibrated per family plus prior baseline."""
    rows = []
    s = bundle.splits
    for name in MODEL_NAMES:
        for version, estimator in [("Raw", bundle.models[name]), ("Calibrated", bundle.calibrated[name])]:
            rows.append({"model": name, "version": version,
                         **metrics_at_threshold(s.y_test, estimator.predict_proba(s.X_test)[:, 1], threshold)})
    rows.append({"model": "Prevalence baseline", "version": "Prior",
                 **metrics_at_threshold(s.y_test, bundle.baseline.predict_proba(s.X_test)[:, 1], threshold)})
    return pd.DataFrame(rows)


def training_diagnostics(bundle: ModelBundle) -> pd.DataFrame:
    """Compare raw base models on reused training and held-out test rows.

    Training metrics are optimistic resubstitution scores, useful only as an
    overfitting diagnostic. They do not evaluate calibration and must not be
    interpreted as independent estimates of generalization performance.
    """
    s = bundle.splits
    rows = []
    for name in MODEL_NAMES:
        model = bundle.models[name]
        train_p = model.predict_proba(s.X_train)[:, 1]
        test_p = model.predict_proba(s.X_test)[:, 1]
        train_auc = float(roc_auc_score(s.y_train, train_p))
        test_auc = float(roc_auc_score(s.y_test, test_p))
        rows.append({
            "model": name, "train_roc_auc": train_auc, "test_roc_auc": test_auc,
            "train_brier_score": float(brier_score_loss(s.y_train, train_p)),
            "test_brier_score": float(brier_score_loss(s.y_test, test_p)),
            "auc_gap": train_auc - test_auc,
        })
    return pd.DataFrame(rows)


def probability_curves(y: pd.Series | np.ndarray, prob: np.ndarray) -> dict[str, np.ndarray]:
    """Full-resolution ROC and precision-recall arrays for plotting."""
    fpr, tpr, roc_thresholds = roc_curve(y, prob)
    precision, recall, pr_thresholds = precision_recall_curve(y, prob)
    return {"fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds,
            "precision": precision, "recall": recall, "pr_thresholds": pr_thresholds}


def reliability_table(y: pd.Series | np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability bins, including empty bins and sample counts.

    Bands are 95% binomial Wilson intervals, not individual
    uncertainty intervals. Empty-bin probabilities/rates remain NaN.
    """
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    probabilities = np.asarray(prob, dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    frame = pd.DataFrame({"observed": np.asarray(y), "probability": probabilities})
    edges = np.linspace(0, 1, n_bins + 1)
    frame["bin"] = pd.cut(frame["probability"], edges, include_lowest=True, labels=False)
    rows = []
    for i in range(n_bins):
        part = frame[frame["bin"] == i]
        count = len(part)
        observed = float(part["observed"].mean()) if count else np.nan
        if count:
            z = 1.96
            denominator = 1 + z * z / count
            center = (observed + z * z / (2 * count)) / denominator
            half = z * np.sqrt(observed * (1 - observed) / count + z * z / (4 * count * count)) / denominator
            low, high = max(0, center - half), min(1, center + half)
        else:
            low, high = np.nan, np.nan
        rows.append({"bin": i, "lower": edges[i], "upper": edges[i + 1], "count": count,
                     "mean_probability": float(part["probability"].mean()) if count else np.nan,
                     "observed_rate": observed,
                     "ci_lower": float(low), "ci_upper": float(high)})
    return pd.DataFrame(rows)


def threshold_sweep(y: pd.Series | np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    """Metrics for fixed hypothetical thresholds; no operating-point selection."""
    return pd.DataFrame([metrics_at_threshold(y, prob, float(t)) for t in np.linspace(.05, .95, 19)])


def suggest_threshold(bundle: ModelBundle, model_name: str, fp_cost: float = 1, fn_cost: float = 5) -> dict:
    """Illustrative cost minimization on calibration rows only.

    The same rows fitted the probability calibrator, so this estimate is
    optimistic and is not an independently validated operating threshold.
    Costs are fictional units, not monetary loss estimates or lending policy.
    """
    if fp_cost < 0 or fn_cost < 0 or fp_cost + fn_cost <= 0:
        raise ValueError("Costs must be nonnegative with at least one positive cost.")
    s = bundle.splits
    table = threshold_sweep(s.y_cal, bundle.calibrated[model_name].predict_proba(s.X_cal)[:, 1])
    table["cost"] = table["fp"] * fp_cost + table["fn"] * fn_cost
    row = table.loc[table["cost"].idxmin()].to_dict()
    row["source"] = "calibration split (optimistic; also used to fit calibrator)"
    return row
