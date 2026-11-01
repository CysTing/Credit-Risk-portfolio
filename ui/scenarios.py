"""Wholly invented profiles. No sample is copied from a UCI customer row."""
from __future__ import annotations

import pandas as pd

from credit_risk.data import BILL_FEATURES, FINANCIAL_FEATURES, PAYMENT_FEATURES, PAY_FEATURES

PRESETS = ("Steady payments", "Recent payment delays", "Low statement balances")


def fictional_profile(name: str) -> pd.DataFrame:
    if name == PRESETS[0]:
        limit, status = 100_000, [0, 0, 0, 0, 0, 0]
        bills, payments = [24000, 26000, 23000, 28000, 25000, 27000], [3500] * 6
    elif name == PRESETS[1]:
        limit, status = 100_000, [2, 2, 1, 0, 0, 0]
        bills, payments = [85000, 82000, 78000, 76000, 73000, 70000], [0, 800, 1200, 3000, 3200, 3500]
    elif name == PRESETS[2]:
        limit, status = 200_000, [-1] * 6
        bills, payments = [6000, 8000, 7000, 5000, 6500, 7500], [6000, 8000, 7000, 5000, 6500, 7500]
    else:
        raise ValueError(f"Unknown fictional preset: {name}")
    row = {"LIMIT_BAL": limit}
    row.update(zip(PAY_FEATURES, status))
    row.update(zip(BILL_FEATURES, bills))
    row.update(zip(PAYMENT_FEATURES, payments))
    return pd.DataFrame([row], columns=FINANCIAL_FEATURES)
