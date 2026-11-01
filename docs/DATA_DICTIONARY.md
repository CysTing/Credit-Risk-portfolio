# Data dictionary

The derivative is based on [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Amounts are **New Taiwan dollars (NT$)**, not US dollars. All financial amounts are recorded historical values; the app's fictional profiles do not refer to actual customers.

## Included fields

| Field | Meaning | Role |
|---|---|---|
| `LIMIT_BAL` | Given credit amount, including individual and supplementary family credit | Predictor |
| `PAY_0` | September 2005 repayment status | Predictor |
| `PAY_2` | August 2005 repayment status | Predictor |
| `PAY_3` | July 2005 repayment status | Predictor |
| `PAY_4` | June 2005 repayment status | Predictor |
| `PAY_5` | May 2005 repayment status | Predictor |
| `PAY_6` | April 2005 repayment status | Predictor |
| `BILL_AMT1` … `BILL_AMT6` | Statement amounts, September through April 2005 | Predictors |
| `PAY_AMT1` … `PAY_AMT6` | Previous payment amounts, September through April 2005 | Predictors |
| `default` | Next-month default payment: `1` yes, `0` no | Target, never an input |

The original naming skips `PAY_1`. `PAY_0` is the most recent repayment status and aligns with `BILL_AMT1` and `PAY_AMT1` in September. Do not shift the histories by interpreting `PAY_2` as the most recent month.

## Repayment status codes

UCI documents `-1` as pay duly; `1` through `8` as delays of one through eight months; `9` as nine months and above. The observed file also contains `-2` and `0`, whose meanings are not specified in the UCI variable description. This project preserves them as **special observed codes** and does not relabel them as a verified repayment state. Values are encoded categorically, keeping these states distinct. Do not assume `-2`, `-1`, and `0` form a meaningful ordered scale.

Positive repayment codes describe the recorded delay category rather than the number of missed payments in the six-month window. Histories can legitimately differ across months.

## Removed fields

| Original field | Reason for exclusion |
|---|---|
| `ID` | Row identifier; no predictive meaning |
| `SEX` | Sensitive demographic attribute |
| `EDUCATION` | Demographic / socioeconomic attribute excluded by project design |
| `MARRIAGE` | Sensitive demographic attribute |
| `AGE` | Sensitive demographic attribute |

These fields are absent from the distributed derivative and feature interface. Their exclusion prevents direct use, not indirect demographic inference or bias.

## Quality and transformation notes

Five additional financial features are constructed inside each model pipeline. All ratios are clipped to `[-10, 10]` by a fixed rule. A nonpositive or missing limit gives a missing limit-based ratio, subsequently imputed using training medians.

| Derived feature | Formula before clipping |
|---|---|
| `latest_utilization` | `BILL_AMT1 / LIMIT_BAL` |
| `mean_utilization` | Mean of six bill amounts divided by `LIMIT_BAL` |
| `latest_payment_ratio` | `PAY_AMT1 / max(abs(BILL_AMT2), 1)` |
| `bill_trend_to_limit` | `(BILL_AMT1 - BILL_AMT6) / LIMIT_BAL` |
| `mean_payment_to_limit` | Mean of six payment amounts divided by `LIMIT_BAL` |

`latest_payment_ratio` is a descriptive ratio of neighboring reporting periods. It is not a reconstruction of contractual dues or a verified repayment coverage measure.

- The loader checks the required financial schema and binary target. Model preprocessing uses an explicit allowlist so extra fields cannot silently become predictors.
- Negative bill balances can reflect credits; they are retained. Payment amounts must be nonnegative for fictional input validation. A strictly positive limit guards division by zero.
- Financial ratios are derived only from supplied financial columns; guarded denominators prevent nonfinite values. Exact formulas are kept next to the transformer in source so documentation and implementation can be reviewed together.
- Training-fitted imputation and scaling belong inside the estimator pipeline. No mean, median, scale, or reference profile is learned from test rows.
- No IDs or historical rows are used as interactive examples. A fictional profile is a complete six-month financial history.
- UCI's published dataset is a historical research resource. Public availability does not make it suitable for contemporary lending deployment.

For original definitions and attribution, consult the [UCI metadata](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and [data README](../data/README.md).
