"""Leakage-free financial feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BILL_FEATURES, FINANCIAL_FEATURES, PAYMENT_FEATURES, PAY_FEATURES

ENGINEERED_FEATURES = [
    "latest_utilization", "mean_utilization", "latest_payment_ratio",
    "bill_trend_to_limit", "mean_payment_to_limit",
]
NUMERIC_FEATURES = ["LIMIT_BAL", *BILL_FEATURES, *PAYMENT_FEATURES, *ENGINEERED_FEATURES]


class FinancialFeatures(TransformerMixin, BaseEstimator):
    """Select financial fields and derive ratios without learning from outcomes.

    Ratios are capped at +/-10 to limit division-by-small-denominator effects.
    This is a fixed preprocessing rule, not an estimate from test data. Raw
    monetary fields remain available and unmodified. Nonpositive/missing credit
    limits produce missing ratios, handled by train-fitted median imputation.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FinancialFeatures":
        self.feature_names_in_ = np.asarray(FINANCIAL_FEATURES, dtype=object)
        self.n_features_in_ = len(FINANCIAL_FEATURES)
        self.transform(X)  # fail early for a malformed training schema
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FINANCIAL_FEATURES) or X.columns.duplicated().any():
            raise ValueError("FinancialFeatures requires exactly the 19 named financial columns.")
        result = X[FINANCIAL_FEATURES].copy()
        limit = result["LIMIT_BAL"].where(result["LIMIT_BAL"] > 0)
        result["latest_utilization"] = result["BILL_AMT1"] / limit
        result["mean_utilization"] = result[BILL_FEATURES].mean(axis=1) / limit
        # PAY_AMT1 and BILL_AMT2 refer to neighbouring reporting periods. This
        # is a descriptive ratio, not a reconstruction of contractual dues.
        result["latest_payment_ratio"] = result["PAY_AMT1"] / result["BILL_AMT2"].abs().clip(lower=1)
        result["bill_trend_to_limit"] = (result["BILL_AMT1"] - result["BILL_AMT6"]) / limit
        result["mean_payment_to_limit"] = result[PAYMENT_FEATURES].mean(axis=1) / limit
        result[ENGINEERED_FEATURES] = result[ENGINEERED_FEATURES].clip(-10, 10)
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(FINANCIAL_FEATURES + ENGINEERED_FEATURES, dtype=object)


def make_preprocessor() -> ColumnTransformer:
    """Fit medians, scaling, and status encodings on training rows only."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        # Preserve -2, -1 and 0 separately: the source does not document all
        # observed codes consistently, so do not invent equivalence meanings.
        ("encoder", OneHotEncoder(
            categories=[list(range(-2, 10)) for _ in PAY_FEATURES],
            handle_unknown="ignore", sparse_output=False,
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("status", categorical, PAY_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
