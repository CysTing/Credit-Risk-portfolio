# Dataset attribution and derivative

**Source:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H). [Dataset page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

The UCI dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Preserve this credit, license link, and the description of modifications when sharing the derivative. No endorsement by the original creator or UCI is implied.

`credit_default.csv.gz` is a compressed derivative of the original Excel workbook. It keeps all 30,000 observations, selects only 19 financial predictors, removes `ID`, `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE`, and renames the original default response to `default`. No original demographic values are distributed in the derivative. Financial values and binary response labels are retained rather than generated.

Rebuild the derivative from the official UCI archive by running `python scripts/download_data.py` from the repository root. See the generated provenance metadata, when present, for file hashes and schema details. The app reads the bundled file locally; it performs no runtime network download.

The dataset describes Taiwan credit card clients and April–September 2005 financial histories. It is an educational research dataset, not a current credit bureau source or a dataset of general loan applications. All interactive examples in the app are fictional. The dataset's CC BY 4.0 license is separate from the project's MIT source-code license.
