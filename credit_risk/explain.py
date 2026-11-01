"""Modest, explicitly scoped model explanations for an educational dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import FINANCIAL_FEATURES, validate_profile
from .modeling import ModelBundle


def logistic_contributions(bundle: ModelBundle, profile: pd.DataFrame) -> pd.DataFrame:
    """Exact raw logistic log-odds terms relative to the zero transformed input.

    Sum(contribution) + intercept equals the raw decision_function, including
    one-hot status terms. These terms do not add up to the sigmoid-calibrated
    probability and are not causal effects or legally valid adverse reasons.
    """
    clean = validate_profile(profile)
    if len(clean) != 1:
        raise ValueError("Explain exactly one fictional profile at a time.")
    pipeline = bundle.models["Logistic regression"]
    engineered = pipeline.named_steps["features"].transform(clean)
    preprocess = pipeline.named_steps["preprocess"]
    transformed = preprocess.transform(engineered)[0]
    classifier = pipeline.named_steps["classifier"]
    result = pd.DataFrame({
        "feature": preprocess.get_feature_names_out(),
        "contribution": transformed * classifier.coef_[0],
    })
    result.attrs["intercept"] = float(classifier.intercept_[0])
    result.attrs["raw_log_odds"] = float(pipeline.decision_function(clean)[0])
    result.attrs["raw_probability"] = float(pipeline.predict_proba(clean)[0, 1])
    return result.sort_values("contribution", key=lambda x: x.abs(), ascending=False).reset_index(drop=True)


def reference_effects(bundle: ModelBundle, profile: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """One-at-a-time substitution of a synthetic training-summary reference.

    Positive change_pp means the original fictional profile has a higher
    calibrated probability than the same profile with just that field replaced
    by its training median (amounts) or mode (status). Values are not additive
    and can break correlations.
    """
    clean = validate_profile(profile)
    if len(clean) != 1:
        raise ValueError("Explain exactly one fictional profile at a time.")
    model = bundle.calibrated[model_name]
    original = float(model.predict_proba(clean)[0, 1])
    # Monetary medians may be fractional even when the source columns contain
    # integers. Use a floating frame before substitution (pandas 3-compatible).
    candidates = pd.concat([clean] * len(FINANCIAL_FEATURES), ignore_index=True).astype(float)
    for i, col in enumerate(FINANCIAL_FEATURES):
        candidates.loc[i, col] = bundle.reference.iloc[0][col]
    substituted = model.predict_proba(candidates)[:, 1]
    result = pd.DataFrame({"feature": FINANCIAL_FEATURES,
                           "change_pp": (original - substituted) * 100,
                           "reference_value": bundle.reference.iloc[0].to_numpy()})
    return result.sort_values("change_pp", key=lambda x: x.abs(), ascending=False).reset_index(drop=True)


def permutation_importance_table(bundle: ModelBundle, model_name: str, max_samples: int = 1000) -> pd.DataFrame:
    """Held-out ROC-AUC decrease after permuting each raw financial field.

    Correlated features may share or hide importance. Negative importances are
    retained. This is descriptive model inspection, not model selection.
    """
    if max_samples < 2:
        raise ValueError("Use at least two held-out samples.")
    s = bundle.splits
    sample = s.X_test.sample(n=min(max_samples, len(s.X_test)), random_state=42)
    result = permutation_importance(
        bundle.calibrated[model_name], sample, s.y_test.loc[sample.index],
        scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1,
    )
    return pd.DataFrame({"feature": FINANCIAL_FEATURES,
                         "importance": result.importances_mean,
                         "std": result.importances_std}).sort_values("importance", ascending=False).reset_index(drop=True)
