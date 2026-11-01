# Validation notes

The project was exercised locally with Python 3.12.14 on Windows, using the pinned application and development dependencies. The tested transitive versions are recorded in `constraints.txt`. Linux CI is configured in `.github/workflows/ci.yml`; a successful remote GitHub Actions run or hosted deployment is not claimed.

## Automated checks

`python -m pytest` passed **45 tests in 11.20 seconds**, with no warnings in the final full run. Coverage includes:

- Strict financial input allowlist and rejected demographic, malformed, missing, infinite, and invalid status inputs.
- Deterministic, disjoint row-index splits and training-only imputation/scaling.
- A fit spy confirming the calibrator receives only the calibration partition.
- Finite probabilities, positive-class metrics, inclusive threshold boundaries, confusion totals, and reliability bin endpoints.
- The exact raw logistic log-odds identity and reference-substitution semantics, including fractional medians.
- Fixed-seed permutation importance and fictional presets absent from the UCI data.
- Raw-model training/test score consistency and the direction of the reported AUC gap.
- Streamlit AppTest on the real bundled data, visiting all four views, switching both models, selecting a fictional preset, and submitting an edited credit limit.

`python -m pip check` reported no broken requirements. All Python source files compiled successfully. The full `scripts/train.py --importance` run completed successfully in approximately 11 seconds and produced the checked-in evaluation, reliability, feature-importance, split, and provenance reports.

## Interface checks

The app was served locally on port 8501 and inspected in a browser. The overview rendered the KPI cards, histogram, confusion matrix, and aggregate repayment chart. Navigation, model comparison, scenario flow, and calibration were checked alongside the AppTest coverage. The global permutation-importance button completed and displayed its chart and table. The theme uses native Streamlit configuration and components, without injected CSS or JavaScript. The narrow preview wraps metric cards and collapses the sidebar; a 1440-pixel desktop check confirmed the four metrics and paired charts remain readable with the sidebar expanded. See `overview.jpg` for the captured desktop view.

## Limits of verification

These are software and local rendering checks, not independent evidence of credit model validity. No current borrower data, fairness audit, production security assessment, out-of-time validation, or real credit decision was involved. Streamlit Community Cloud deployment requires the owner's GitHub and Cloud account; the repository includes instructions but has not been published by this build.
