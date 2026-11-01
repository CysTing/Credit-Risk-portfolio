"""Dataset schema, provenance, and strict financial-only input validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "credit_default.csv.gz"
TARGET = "default"
PAY_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]
FINANCIAL_FEATURES = ["LIMIT_BAL", *PAY_FEATURES, *BILL_FEATURES, *PAYMENT_FEATURES]
EXCLUDED_FEATURES = ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
MONTHS = ["September", "August", "July", "June", "May", "April"]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled UCI-derived financial data; never fetch on app startup.

    Demographics and identifiers must already have been removed. The public
    source contains 30,000 rows, but smaller fixtures are accepted for tests.
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {path}. Run python scripts/download_data.py first."
        )
    frame = pd.read_csv(path)
    expected = set(FINANCIAL_FEATURES + [TARGET])
    if set(frame.columns) != expected:
        raise ValueError("Dataset must contain exactly 19 financial features and default.")
    if frame.empty or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("The default target must be nonmissing and binary (0 or 1).")
    for col in FINANCIAL_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if np.isinf(frame[FINANCIAL_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite financial values are invalid.")
    frame[TARGET] = frame[TARGET].astype(int)
    return frame[FINANCIAL_FEATURES + [TARGET]]


def validate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric fictional inputs with a strict feature allowlist.

    Extra columns (including demographic attributes), missing values, infinite
    numbers, unsupported payment codes, and negative payments are rejected.
    Negative statement balances are valid credits and remain unchanged.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Provide at least one fictional financial profile as a DataFrame.")
    if set(frame.columns) != set(FINANCIAL_FEATURES) or frame.columns.duplicated().any():
        raise ValueError("Provide exactly the 19 financial features; demographic fields are excluded.")
    try:
        clean = frame[FINANCIAL_FEATURES].apply(pd.to_numeric, errors="raise").copy()
    except (ValueError, TypeError) as exc:
        raise ValueError("All financial inputs must be numeric.") from exc
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Financial inputs must be finite and nonmissing.")
    if (clean["LIMIT_BAL"] <= 0).any():
        raise ValueError("Credit limit must be positive.")
    if (clean[PAYMENT_FEATURES] < 0).any().any():
        raise ValueError("Payments must be nonnegative.")
    if not clean[PAY_FEATURES].isin(range(-2, 10)).all().all():
        raise ValueError("Repayment status must be an integer UCI code from -2 through 9.")
    return clean
