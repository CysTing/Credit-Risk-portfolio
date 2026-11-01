"""Fictional demo profiles never require demographic inputs or a real row."""

import pytest

from credit_risk.data import FINANCIAL_FEATURES, load_dataset, validate_profile
from ui.scenarios import PRESETS, fictional_profile


@pytest.mark.parametrize("name", PRESETS)
def test_demo_presets_are_valid_financial_profiles_and_not_source_rows(name):
    profile = fictional_profile(name)
    clean = validate_profile(profile)
    assert clean.shape == (1, 19)
    assert list(clean.columns) == FINANCIAL_FEATURES
    actual_data = load_dataset()[FINANCIAL_FEATURES]
    assert not actual_data.eq(clean.iloc[0]).all(axis=1).any()


def test_unknown_preset_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown fictional preset"):
        fictional_profile("real customer")
