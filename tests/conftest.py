"""Deterministic fictional financial records for fast software tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from credit_risk.data import BILL_FEATURES, FINANCIAL_FEATURES, PAYMENT_FEATURES, PAY_FEATURES, TARGET
from credit_risk.modeling import train_models


@pytest.fixture(scope="session")
def fictional_data() -> pd.DataFrame:
    """Generate data for invariants, not a surrogate UCI performance benchmark."""
    rng = np.random.default_rng(2026)
    n = 480
    frame = pd.DataFrame(index=np.arange(1000, 1000 + n))
    frame["LIMIT_BAL"] = rng.integers(1, 26, n) * 10_000
    for col in PAY_FEATURES:
        frame[col] = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[.1, .2, .3, .2, .15, .05])
    for col in BILL_FEATURES:
        frame[col] = rng.integers(-1000, 150_000, n)
    for col in PAYMENT_FEATURES:
        frame[col] = rng.integers(0, 20_000, n)
    probability = .08 + .48 * (frame["PAY_0"] >= 2) + .15 * (frame["BILL_AMT1"] > frame["LIMIT_BAL"])
    frame[TARGET] = (rng.random(n) < probability).astype(int)
    return frame[FINANCIAL_FEATURES + [TARGET]]


@pytest.fixture(scope="session")
def fitted_bundle(fictional_data):
    return train_models(fictional_data, n_estimators=8)


@pytest.fixture
def fictional_profile(fictional_data):
    # The entire fixture is generated; it contains no historical customer row.
    return fictional_data.drop(columns=[TARGET]).iloc[[0]].copy()
