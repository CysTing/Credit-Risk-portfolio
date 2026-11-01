"""Schema and fictional profile constraints."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import EXCLUDED_FEATURES, FINANCIAL_FEATURES, TARGET, load_dataset, validate_profile


def test_bundled_derivative_is_complete_and_financial_only():
    data = load_dataset()
    assert data.shape == (30_000, 20)
    assert list(data.columns) == FINANCIAL_FEATURES + [TARGET]
    assert not set(EXCLUDED_FEATURES).intersection(data.columns)
    assert data[TARGET].value_counts().to_dict() == {0: 23_364, 1: 6636}


def test_loader_rejects_demographic_column(tmp_path, fictional_data):
    path = tmp_path / "invalid.csv"
    fictional_data.assign(SEX=1).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly 19 financial"):
        load_dataset(path)


@pytest.mark.parametrize("value", [np.nan, 2, -1])
def test_loader_rejects_invalid_target(tmp_path, fictional_data, value):
    data = fictional_data.copy()
    data.loc[data.index[0], TARGET] = value
    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_dataset(path)


def test_profile_reorders_columns_and_preserves_bill_credits(fictional_profile):
    fictional_profile.loc[:, "BILL_AMT1"] = -250
    clean = validate_profile(fictional_profile[list(reversed(FINANCIAL_FEATURES))])
    assert list(clean.columns) == FINANCIAL_FEATURES
    assert clean["BILL_AMT1"].iloc[0] == -250


@pytest.mark.parametrize("column,value", [
    ("LIMIT_BAL", 0), ("LIMIT_BAL", -10), ("LIMIT_BAL", np.inf),
    ("PAY_AMT1", -1), ("BILL_AMT1", np.nan), ("PAY_0", 1.5),
    ("PAY_0", 10), ("BILL_AMT1", "not-a-number"),
])
def test_profile_rejects_invalid_values(fictional_profile, column, value):
    invalid = fictional_profile.astype(object)
    invalid.loc[:, column] = value
    with pytest.raises(ValueError):
        validate_profile(invalid)


@pytest.mark.parametrize("extra", ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE", "default"])
def test_profile_rejects_nonfinancial_fields(fictional_profile, extra):
    with pytest.raises(ValueError, match="exactly the 19 financial"):
        validate_profile(fictional_profile.assign(**{extra: 1}))


def test_profile_rejects_missing_or_duplicate_column(fictional_profile):
    with pytest.raises(ValueError):
        validate_profile(fictional_profile.drop(columns="PAY_0"))
    duplicate = pd.concat([fictional_profile, fictional_profile[["PAY_0"]]], axis=1)
    with pytest.raises(ValueError):
        validate_profile(duplicate)
