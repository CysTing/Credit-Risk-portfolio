# Streamlit Community Cloud configuration

These instructions configure an existing project for [Streamlit Community Cloud](https://share.streamlit.io/). The supplied project is deployment-ready source; no live deployment is included.

## App settings

In Community Cloud, choose **Create app**, select the existing repository and branch, and use these settings:

| Setting | Value |
| --- | --- |
| Main file path | `app.py`, relative to the project root |
| Python version | **3.12**, in **Advanced settings** |
| Dependency file | `requirements.txt` |
| Dependency constraints | `constraints.txt`, referenced by `requirements.txt` |
| Secrets | None required |

Keep the complete project together, including `credit_risk/`, `ui/`, `.streamlit/config.toml`, and the bundled financial-only `data/credit_default.csv.gz`. The app reads this local dataset and does not download UCI data on startup. If the project is nested inside a larger repository, select its relative `app.py` path and check the file organization instructions linked from Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Startup and verification

Deploy the configured app and inspect the build logs. On its first start, the app fits both classifiers and their probability calibrators, then caches them in process memory. A restart or wake from sleep rebuilds that cache, so the first page can take longer to load.

After launch, check all four views, switch between both models, move the teaching threshold, and submit a fictional scenario. Confirm that charts render and the educational-use notice remains visible. No account credentials or customer data are needed for those checks.

If startup fails, verify Python 3.12, dependency and constraint files, case-sensitive paths, and the compressed dataset's presence. Use the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/concepts/dependencies) for platform troubleshooting.

This application is educational and must not be used for real credit approval or decisions about real people. Use invented inputs in the public demonstration.
