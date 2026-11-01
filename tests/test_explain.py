"""Explanations must describe the quantity they actually compute."""

import numpy as np
import pytest

from credit_risk.data import FINANCIAL_FEATURES
from credit_risk.explain import logistic_contributions, permutation_importance_table, reference_effects


def test_logistic_terms_reconstruct_raw_log_odds(fitted_bundle, fictional_profile):
    terms = logistic_contributions(fitted_bundle, fictional_profile)
    reconstructed = terms["contribution"].sum() + terms.attrs["intercept"]
    expected = fitted_bundle.models["Logistic regression"].decision_function(fictional_profile)[0]
    assert reconstructed == pytest.approx(expected, abs=1e-10)
    assert terms.attrs["raw_log_odds"] == pytest.approx(expected)
    assert terms.attrs["raw_probability"] == pytest.approx(1 / (1 + np.exp(-expected)))
    assert not any(feature in {"ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"} for feature in terms["feature"])


@pytest.mark.filterwarnings("error::FutureWarning")
def test_reference_effect_matches_single_input_substitution(fitted_bundle, fictional_profile):
    effects = reference_effects(fitted_bundle, fictional_profile, "Random forest")
    assert set(effects["feature"]) == set(FINANCIAL_FEATURES)
    model = fitted_bundle.calibrated["Random forest"]
    changed = fictional_profile.copy()
    changed.loc[:, "LIMIT_BAL"] = fitted_bundle.reference["LIMIT_BAL"].iloc[0]
    expected = (model.predict_proba(fictional_profile)[0, 1] - model.predict_proba(changed)[0, 1]) * 100
    assert effects.set_index("feature").loc["LIMIT_BAL", "change_pp"] == pytest.approx(expected)


def test_importance_uses_financial_raw_fields_and_reports_variation(fitted_bundle):
    table = permutation_importance_table(fitted_bundle, "Logistic regression", max_samples=64)
    assert set(table["feature"]) == set(FINANCIAL_FEATURES)
    assert set(table.columns) == {"feature", "importance", "std"}
    assert np.isfinite(table[["importance", "std"]]).all().all()
    assert (table["std"] >= 0).all()


def test_explanations_reject_multiple_profiles(fitted_bundle, fictional_data):
    profiles = fictional_data[FINANCIAL_FEATURES].iloc[:2]
    with pytest.raises(ValueError, match="exactly one"):
        logistic_contributions(fitted_bundle, profiles)
    with pytest.raises(ValueError, match="exactly one"):
        reference_effects(fitted_bundle, profiles, "Random forest")
