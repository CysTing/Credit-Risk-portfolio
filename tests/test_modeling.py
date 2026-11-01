"""Leakage boundaries, calibration protocol, and metric semantics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from credit_risk.data import FINANCIAL_FEATURES, PAY_FEATURES, TARGET
from credit_risk.features import ENGINEERED_FEATURES, FinancialFeatures, NUMERIC_FEATURES
from credit_risk.modeling import (
    MODEL_NAMES, build_pipeline, evaluate_models, metrics_at_threshold,
    reliability_table, split_data, suggest_threshold, train_models, training_diagnostics,
)


def test_splits_are_disjoint_deterministic_and_stratified(fictional_data):
    first, second = split_data(fictional_data), split_data(fictional_data)
    indices = [set(first.X_train.index), set(first.X_cal.index), set(first.X_test.index)]
    assert [len(x) for x in indices] == [288, 96, 96]
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
    assert set.union(*indices) == set(fictional_data.index)
    for name in ("train", "cal", "test"):
        pd.testing.assert_frame_equal(getattr(first, f"X_{name}"), getattr(second, f"X_{name}"))
        pd.testing.assert_index_equal(getattr(first, f"X_{name}").index, getattr(first, f"y_{name}").index)
        assert abs(getattr(first, f"y_{name}").mean() - fictional_data[TARGET].mean()) < .02


def test_training_statistics_use_training_rows_only(fitted_bundle):
    split = fitted_bundle.splits
    for model in fitted_bundle.models.values():
        engineered = model.named_steps["features"].transform(split.X_train)
        numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
        expected_medians = engineered[NUMERIC_FEATURES].median().to_numpy()
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected_medians)
        assert numeric.named_steps["scaler"].n_samples_seen_ == len(split.X_train)
        np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, engineered[NUMERIC_FEATURES].mean())
    expected_reference = split.X_train.median().to_frame().T[FINANCIAL_FEATURES]
    for col in PAY_FEATURES:
        expected_reference.loc[expected_reference.index[0], col] = split.X_train[col].mode().iloc[0]
    pd.testing.assert_frame_equal(fitted_bundle.reference, expected_reference)


def test_heldout_transform_does_not_refit_preprocessing(fictional_data):
    split = split_data(fictional_data)
    pipeline = build_pipeline("Logistic regression").fit(split.X_train, split.y_train)
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    original = numeric.named_steps["imputer"].statistics_.copy()
    extreme = split.X_test.copy()
    extreme["LIMIT_BAL"] = 1e12
    extreme["BILL_AMT1"] = np.nan
    assert np.isfinite(pipeline.predict_proba(extreme)).all()
    np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_, original)


def test_calibrator_receives_only_disjoint_calibration_rows(monkeypatch, fictional_data):
    received = []
    actual_fit = CalibratedClassifierCV.fit

    def spy_fit(self, X, y, **kwargs):
        received.append((X.index.copy(), y.index.copy()))
        return actual_fit(self, X, y, **kwargs)

    monkeypatch.setattr(CalibratedClassifierCV, "fit", spy_fit)
    bundle = train_models(fictional_data, n_estimators=3)
    assert len(received) == 2
    for X_index, y_index in received:
        pd.testing.assert_index_equal(X_index, bundle.splits.X_cal.index)
        pd.testing.assert_index_equal(y_index, bundle.splits.y_cal.index)
    for name in MODEL_NAMES:
        assert bundle.calibrated[name].estimator.estimator is bundle.models[name]
        scaler = bundle.models[name].named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        assert scaler.n_samples_seen_ == len(bundle.splits.X_train)


def test_probabilities_and_baseline_are_valid(fitted_bundle):
    for estimator in [*fitted_bundle.models.values(), *fitted_bundle.calibrated.values(), fitted_bundle.baseline]:
        prob = estimator.predict_proba(fitted_bundle.splits.X_test)
        assert prob.shape == (96, 2)
        assert np.isfinite(prob).all()
        assert ((0 <= prob) & (prob <= 1)).all()
        np.testing.assert_allclose(prob.sum(axis=1), 1)
    baseline = fitted_bundle.baseline.predict_proba(fitted_bundle.splits.X_test)[:, 1]
    np.testing.assert_allclose(baseline, fitted_bundle.splits.y_train.mean())


def test_feature_engineering_guards_small_denominators(fictional_profile):
    data = fictional_profile.copy()
    data.loc[:, "LIMIT_BAL"] = 0
    data.loc[:, "BILL_AMT2"] = 0
    data.loc[:, "PAY_AMT1"] = 1e9
    result = FinancialFeatures().fit_transform(data)
    assert not np.isinf(result.to_numpy(dtype=float)).any()
    assert result["latest_payment_ratio"].iloc[0] == 10
    assert result["latest_utilization"].isna().all()
    finite_ratios = result[ENGINEERED_FEATURES].stack()
    assert finite_ratios.between(-10, 10).all()


def test_special_payment_codes_have_separate_categories(fitted_bundle):
    encoder = fitted_bundle.models["Logistic regression"].named_steps["preprocess"].named_transformers_["status"].named_steps["encoder"]
    for categories in encoder.categories_:
        assert {-2, -1, 0}.issubset(categories)
        assert len(categories) == 12


def test_metrics_use_positive_default_class_and_inclusive_threshold():
    y = np.array([0, 0, 1, 1])
    metrics = metrics_at_threshold(y, np.array([.1, .6, .5, .9]), .5)
    assert {key: metrics[key] for key in ("tn", "fp", "fn", "tp")} == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert metrics["recall"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["flag_rate"] == .75
    assert sum(metrics[key] for key in ("tn", "fp", "fn", "tp")) == metrics["n"]


@pytest.mark.parametrize("prob,threshold", [([.1, np.nan], .5), ([.1, 1.1], .5), ([.1, .8], -1)])
def test_metrics_reject_invalid_probabilities_or_thresholds(prob, threshold):
    with pytest.raises(ValueError):
        metrics_at_threshold([0, 1], np.array(prob), threshold)


def test_reliability_bins_keep_empty_bins_and_endpoints():
    result = reliability_table([0, 0, 1, 1], [0, .1, .9, 1], n_bins=10)
    assert len(result) == 10
    assert result["count"].sum() == 4
    assert result.loc[0, "count"] == 2
    assert result.loc[9, "count"] == 1
    assert result.loc[result["count"] == 0, "observed_rate"].isna().all()


def test_evaluation_reports_both_versions_and_prevalence_baseline(fitted_bundle):
    report = evaluate_models(fitted_bundle)
    assert len(report) == 5
    assert set(report["version"]) == {"Raw", "Calibrated", "Prior"}
    assert (report[["tn", "fp", "fn", "tp"]].sum(axis=1) == 96).all()


def test_training_diagnostics_match_raw_scores_and_auc_gap(fitted_bundle):
    report = training_diagnostics(fitted_bundle).set_index("model")
    assert set(report.index) == set(MODEL_NAMES)
    assert set(report.columns) == {
        "train_roc_auc", "test_roc_auc", "train_brier_score", "test_brier_score", "auc_gap",
    }
    split = fitted_bundle.splits
    for name in MODEL_NAMES:
        raw = fitted_bundle.models[name]
        train_prob = raw.predict_proba(split.X_train)[:, 1]
        test_prob = raw.predict_proba(split.X_test)[:, 1]
        train_auc = roc_auc_score(split.y_train, train_prob)
        test_auc = roc_auc_score(split.y_test, test_prob)
        assert report.loc[name, "train_roc_auc"] == pytest.approx(train_auc)
        assert report.loc[name, "test_roc_auc"] == pytest.approx(test_auc)
        assert report.loc[name, "train_brier_score"] == pytest.approx(brier_score_loss(split.y_train, train_prob))
        assert report.loc[name, "test_brier_score"] == pytest.approx(brier_score_loss(split.y_test, test_prob))
        assert report.loc[name, "auc_gap"] == pytest.approx(train_auc - test_auc)


def test_threshold_cost_demo_does_not_use_test_predictions(fitted_bundle, monkeypatch):
    model = fitted_bundle.calibrated["Random forest"]
    actual_predict = model.predict_proba
    seen = []

    def spy_predict(X):
        seen.append(X.index.copy())
        return actual_predict(X)

    monkeypatch.setattr(model, "predict_proba", spy_predict)
    result = suggest_threshold(fitted_bundle, "Random forest", fp_cost=1, fn_cost=5)
    assert len(seen) == 1
    pd.testing.assert_index_equal(seen[0], fitted_bundle.splits.X_cal.index)
    assert "optimistic" in result["source"]
    assert .05 <= result["threshold"] <= .95
