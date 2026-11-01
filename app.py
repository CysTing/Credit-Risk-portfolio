"""LendLens: an educational, financial-only credit default modeling lab.

Run from the repository root with: python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_risk.data import (
    BILL_FEATURES, DEFAULT_DATA_PATH, FINANCIAL_FEATURES, MONTHS,
    PAYMENT_FEATURES, PAY_FEATURES, load_dataset, validate_profile,
)
from credit_risk.explain import (
    logistic_contributions, permutation_importance_table, reference_effects,
)
from credit_risk.modeling import (
    MODEL_NAMES, ModelBundle, evaluate_models, metrics_at_threshold,
    probability_curves, reliability_table, threshold_sweep, train_models,
    training_diagnostics,
)
from ui.charts import (
    AMBER, CORAL, NAVY, TEAL, binary_matrix, cohort_rates, horizontal_effects,
    polish, probability_histogram,
)
from ui.scenarios import PRESETS, fictional_profile

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="LendLens · Credit Risk Lab", page_icon="◈", layout="wide")


@st.cache_resource(show_spinner=False)
def fitted_experiment(data_digest: str) -> ModelBundle:
    """Rebuild only when the bundled data changes or the process restarts."""
    return train_models(load_dataset())


@st.cache_data(show_spinner=False)
def test_probabilities(data_digest: str, _bundle: ModelBundle) -> dict[str, np.ndarray]:
    return {name: model.predict_proba(_bundle.splits.X_test)[:, 1]
            for name, model in _bundle.calibrated.items()}


@st.cache_data(show_spinner=False)
def global_importance(data_digest: str, model_name: str, _bundle: ModelBundle) -> pd.DataFrame:
    return permutation_importance_table(_bundle, model_name)


@st.cache_data(show_spinner=False)
def train_test_diagnostics(data_digest: str, _bundle: ModelBundle) -> pd.DataFrame:
    return training_diagnostics(_bundle)


def section(title: str, description: str) -> None:
    st.subheader(title)
    st.caption(description)


def plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def metric_row(items: list[tuple[str, str, str]]) -> None:
    with st.container(horizontal=True, key="kpis"):
        for label, value, help_text in items:
            st.metric(label, value, help=help_text, border=True, width="stretch")


def overview(bundle: ModelBundle, probabilities: dict, name: str, threshold: float) -> None:
    y, p = bundle.splits.y_test, probabilities[name]
    m = metrics_at_threshold(y, p, threshold)
    st.title("Credit risk, made visible.")
    st.markdown("Explore how financial history becomes an estimated probability of next-month credit card default. Compare models, question their confidence, and test a fictional scenario.")
    with st.container(horizontal=True, gap="small"):
        st.badge("30,000 historical records", icon=":material/database:", color="blue")
        st.badge("Financial features only", icon=":material/shield:", color="green")
        st.badge("Educational experiment", icon=":material/school:", color="orange")
    metric_row([
        ("ROC AUC", f"{m['roc_auc']:.3f}", f"{name}, calibrated, final test set. Higher is better; 0.5 is chance ranking."),
        ("Average precision", f"{m['average_precision']:.3f}", "Precision-recall summary. Compare with the test default prevalence."),
        ("Brier score", f"{m['brier_score']:.3f}", "Mean squared probability error, 0 to 1. Lower is better."),
        ("Observed default rate", f"{y.mean():.1%}", f"{len(y):,} held-out test records; this is not a forecast for today's customers."),
    ])
    st.caption(f"Selected: {name} · Sigmoid calibrated · Final test n={len(y):,} · Threshold {threshold:.0%}")
    left, right = st.columns([1.35, 1], gap="large")
    with left, st.container(border=True):
        section("Where the predictions overlap", "Final test set · calibrated probabilities, grouped by the observed outcome")
        plot(probability_histogram(y, p, threshold), "overview_hist")
        st.caption("Overlapping distributions show why a probability is not a certain outcome.")
    with right, st.container(border=True):
        section("A threshold changes the story", f"Final test set · teaching threshold = {threshold:.0%}")
        plot(binary_matrix(y, p, threshold), "overview_cm")
        st.caption(f"Recall {m['recall']:.1%} · Precision {m['precision']:.1%} · F1 {m['f1']:.3f}")
    left, right = st.columns([1.35, 1], gap="large")
    with left, st.container(border=True):
        section("Repayment history in context", "Descriptive view of all 30,000 records · association, not a causal effect")
        plot(cohort_rates(bundle.dataset), "cohort")
        st.caption("−1: paid duly; 1–9: months delayed. Original UCI notes do not define −2 or 0; they remain distinct categories.")
    with right, st.container(border=True):
        section("One experiment. Three separate roles.", "Stratified random splits · seed 42 · no oversampling")
        for label, split, description in [
            ("01 / Fit", bundle.splits.y_train, "Preprocessing and both base classifiers"),
            ("02 / Calibrate", bundle.splits.y_cal, "A separate sigmoid mapping for each classifier"),
            ("03 / Evaluate", bundle.splits.y_test, "Final, held-out comparison and diagnostics"),
        ]:
            st.markdown(f"**{label} — {len(split):,} records**")
            st.caption(description)
        st.info("Accuracy alone can hide missed defaults. Read recall, precision, and probability error together.", icon="ℹ️")


def model_lab(bundle: ModelBundle, probabilities: dict, name: str, threshold: float, digest: str) -> None:
    section("Model laboratory", "Compare the evidence behind each model. Every chart below uses the final held-out test set.")
    metrics = evaluate_models(bundle, threshold)
    columns = ["model", "version", "roc_auc", "average_precision", "brier_score", "log_loss", "precision", "recall", "f1"]
    st.dataframe(metrics[columns], hide_index=True, width="stretch", column_config={
        col: st.column_config.NumberColumn(col.replace("_", " ").title(), format="%.3f")
        for col in columns[2:]
    })
    st.caption("ROC AUC / average precision: higher is better. Brier / log loss: lower is better. Prior baseline predicts training prevalence for everyone. No model is automatically selected from test results.")
    st.download_button("Download evaluation table", metrics.to_csv(index=False), "heldout_metrics.csv", "text/csv")
    with st.expander("Compare training and test performance"):
        st.caption("Raw base models only. Training scores reuse fitted rows and are optimistic; the gap is an overfitting diagnostic, not a second independent evaluation.")
        diagnostics = train_test_diagnostics(digest, bundle)
        st.dataframe(diagnostics, hide_index=True, width="stretch", column_config={
            col: st.column_config.NumberColumn(col.replace("_", " ").title(), format="%.4f")
            for col in diagnostics.columns if col != "model"
        })
    discrimination, calibration, tradeoffs, explanations = st.tabs([
        "Discrimination", "Calibration", "Threshold trade-offs", "Model explanations",
    ])
    y = bundle.splits.y_test
    with discrimination:
        left, right = st.columns(2, gap="large")
        roc, pr = go.Figure(), go.Figure()
        for model_name, color in zip(MODEL_NAMES, [TEAL, NAVY]):
            c = probability_curves(y, probabilities[model_name])
            roc.add_scatter(x=c["fpr"], y=c["tpr"], name=model_name, line_color=color)
            pr.add_scatter(x=c["recall"], y=c["precision"], name=model_name, line_color=color)
        roc.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="#9baec0", dash="dot"))
        pr.add_hline(y=float(y.mean()), line_color=AMBER, line_dash="dot")
        roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate / recall")
        pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        for fig in [roc, pr]:
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        with left, st.container(border=True):
            section("ROC curve", "How well does the model rank default versus no-default observations?")
            plot(polish(roc, 370), "roc")
        with right, st.container(border=True):
            section("Precision–recall curve", f"Dotted baseline = test prevalence ({y.mean():.1%})")
            plot(polish(pr, 370), "pr")
    with calibration:
        left, right = st.columns([1.3, 1], gap="large")
        raw_p = bundle.models[name].predict_proba(bundle.splits.X_test)[:, 1]
        fig = go.Figure()
        for label, prob, color in [("Raw", raw_p, NAVY), ("Calibrated", probabilities[name], TEAL)]:
            bins = reliability_table(y, prob)
            nonempty = bins[bins["count"] > 0]
            fig.add_scatter(x=nonempty["mean_probability"], y=nonempty["observed_rate"],
                            mode="lines+markers", name=label, line_color=color,
                            customdata=nonempty[["count"]],
                            hovertemplate="Mean estimate %{x:.1%}<br>Observed rate %{y:.1%}<br>n=%{customdata[0]:,}<extra>%{fullData.name}</extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(dash="dot", color=AMBER))
        fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed default rate")
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        with left, st.container(border=True):
            section(f"Reliability · {name}", "10 equal-width bins · only nonempty bins are plotted")
            plot(polish(fig, 390), "reliability")
        with right:
            st.markdown("**Does 30% mean about 3 in 10?**")
            st.write("Calibration compares estimated probabilities with observed rates across groups. The sigmoid mapping is fitted on 6,000 separate calibration records, never on the test set.")
            st.write("Calibration may improve probability error without improving ranking. It can also worsen a metric; inspect the raw and calibrated results above.")
            bins = reliability_table(y, probabilities[name])
            st.dataframe(bins[["lower", "upper", "count", "mean_probability", "observed_rate"]], hide_index=True,
                         column_config={col: st.column_config.NumberColumn(format="%.3f") for col in ["lower", "upper", "mean_probability", "observed_rate"]})
            st.caption("Small bins are noisy; these group rates do not provide confidence intervals for a person's outcome.")
    with tradeoffs:
        st.markdown("**Move the teaching threshold in the sidebar.** A higher threshold usually flags fewer observations and misses more observed defaults.")
        table = threshold_sweep(y, probabilities[name])
        fig = go.Figure()
        for metric, color in [("precision", TEAL), ("recall", CORAL), ("flag_rate", NAVY)]:
            fig.add_scatter(x=table["threshold"], y=table[metric], name=metric.replace("_", " ").title(), line_color=color)
        fig.add_vline(x=threshold, line_dash="dash", line_color=AMBER)
        fig.update_layout(xaxis_title="Hypothetical probability threshold", yaxis_title="Fraction of observations")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        plot(polish(fig, 340), "thresholds")
        st.warning("These are descriptive test diagnostics, not a recommended operating threshold. Repeatedly tuning against this chart would invalidate the test set as an independent evaluation.")
        plot(binary_matrix(y, probabilities[name], threshold), "lab_cm")
    with explanations:
        st.write("Permutation importance measures the decrease in test ROC AUC when one financial field is shuffled. Correlated fields can share or hide importance; the chart does not show causes of default.")
        if st.button("Calculate global permutation importance", type="primary"):
            with st.spinner("Inspecting a fixed 1,000-record test sample, 3 repeats per field…"):
                importance = global_importance(digest, name, bundle)
            st.session_state["importance_result"] = (name, importance)
        saved = st.session_state.get("importance_result")
        if saved is not None and saved[0] == name:
            importance = saved[1]
            view = importance.sort_values("importance").tail(12)
            fig = go.Figure(go.Bar(x=view["importance"], y=view["feature"], orientation="h", marker_color=TEAL,
                                   error_x=dict(type="data", array=view["std"])))
            fig.update_layout(xaxis_title="Mean decrease in ROC AUC (error bars: repeat standard deviation)")
            plot(polish(fig, 420), "permutation")
            st.dataframe(importance, hide_index=True)
        st.caption("The Scenario studio also provides local sensitivity checks and exact raw logistic log-odds terms.")


def scenario_studio(bundle: ModelBundle, name: str, threshold: float) -> None:
    section("Scenario studio", "A fictional financial profile, a model estimate, and a closer look at what changes it.")
    st.warning("Use invented numbers only. No names, customer records, or demographic attributes. This demonstration cannot make a lending decision.", icon=":material/science:")
    preset = st.selectbox("Start from a fictional profile", PRESETS)
    default = fictional_profile(preset)
    if st.session_state.get("active_preset") != preset:
        st.session_state["active_preset"] = preset
        st.session_state["fictional_profile"] = default
    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("**01 / Edit the financial history**")
        with st.form(f"scenario_{preset}"):
            limit = st.number_input("Credit limit (NT$)", 10000, 1000000, int(default["LIMIT_BAL"].iloc[0]), 10000)
            monthly = pd.DataFrame({
                "Month": MONTHS, "Status code": default[PAY_FEATURES].iloc[0].to_numpy(dtype=int),
                "Statement (NT$)": default[BILL_FEATURES].iloc[0].to_numpy(dtype=int),
                "Paid (NT$)": default[PAYMENT_FEATURES].iloc[0].to_numpy(dtype=int),
            })
            edited = st.data_editor(monthly, hide_index=True, width="stretch", num_rows="fixed", disabled=["Month"],
                column_config={
                    "Status code": st.column_config.NumberColumn(min_value=-2, max_value=9, step=1, required=True),
                    "Statement (NT$)": st.column_config.NumberColumn(min_value=-200000, max_value=2000000, step=500, required=True),
                    "Paid (NT$)": st.column_config.NumberColumn(min_value=0, max_value=2000000, step=500, required=True),
                }, key=f"history_{preset}")
            st.caption("September → April 2005, newest first. Negative statements represent credit balances. −1: paid duly; 1–9: delay in months. −2 and 0 are distinct, undocumented source codes.")
            submitted = st.form_submit_button("Explore this fictional scenario", type="primary", width="stretch")
            if submitted:
                row = {"LIMIT_BAL": limit}
                row.update(zip(PAY_FEATURES, edited["Status code"]))
                row.update(zip(BILL_FEATURES, edited["Statement (NT$)"]))
                row.update(zip(PAYMENT_FEATURES, edited["Paid (NT$)"]))
                try:
                    st.session_state["fictional_profile"] = validate_profile(pd.DataFrame([row]))
                except ValueError as exc:
                    st.error(str(exc))
        st.caption("Results reflect the last submitted values, or the loaded preset. Apply edits to recalculate.")
        with st.expander("Inspect the submitted fictional inputs"):
            st.dataframe(st.session_state["fictional_profile"].T.rename(columns={0: "Value"}))
    profile = st.session_state["fictional_profile"]
    probability = float(bundle.calibrated[name].predict_proba(profile)[0, 1])
    with right:
        st.markdown("**02 / Interpret the estimate**")
        with st.container(border=True):
            st.badge("Calibrated model estimate", color="blue", icon=":material/analytics:")
            st.caption("Estimated next-month default probability")
            st.title(f"{probability:.1%}")
            st.markdown(f"**{name}** · sigmoid calibrated")
            st.caption("This is a model estimate for an invented profile, not an approval decision or a certain outcome.")
        st.caption(f"Teaching comparison: {'above or equal to' if probability >= threshold else 'below'} the {threshold:.0%} threshold. No action is assigned.")
        comparison = pd.DataFrame({"Model": MODEL_NAMES, "Estimated probability": [
            float(bundle.calibrated[model_name].predict_proba(profile)[0, 1]) for model_name in MODEL_NAMES
        ]})
        st.dataframe(comparison, hide_index=True, width="stretch", column_config={
            "Estimated probability": st.column_config.NumberColumn(format="percent", min_value=0, max_value=1)
        })
        outside = [col for col in FINANCIAL_FEATURES if profile[col].iloc[0] < bundle.splits.X_train[col].min()
                   or profile[col].iloc[0] > bundle.splits.X_train[col].max()]
        if outside:
            st.warning("Outside training range: " + ", ".join(outside) + ". Estimates may be unreliable.")
        st.info("An estimated 20% probability is not a credit score. It does not measure affordability, expected loss, or loan eligibility.")
    section("What is the model responding to?", "Local explanations for the submitted fictional profile")
    effects = reference_effects(bundle, profile, name)
    lcol, rcol = st.columns([1.3, 1], gap="large")
    with lcol, st.container(border=True):
        plot(horizontal_effects(effects, "change_pp", "Original minus substituted estimate (percentage points)"), "local_effects")
    with rcol:
        st.markdown("**A one-field sensitivity check**")
        st.write("For each bar, replace one input with its training reference (median for amounts, most frequent category for statuses), keep the other inputs fixed, and predict again. Positive bars mean the original scenario had a higher estimated probability.")
        st.write("These changes are not additive. Replacements can create unrealistic combinations, and they do not predict what would happen if someone changed their behavior.")
        st.caption("Red = higher original estimate. Teal = lower original estimate. This is not a list of lending reasons or financial advice.")
        with st.expander("See every reference substitution"):
            st.dataframe(effects, hide_index=True)
    with st.expander("Inspect exact raw logistic regression terms"):
        terms = logistic_contributions(bundle, profile)
        st.write("The raw logistic model adds an intercept and feature contributions in log-odds space. These terms explain the uncalibrated logistic model, even when Random forest is selected above.")
        st.caption(f"Intercept {terms.attrs['intercept']:.4f} + all contributions {terms['contribution'].sum():.4f} = raw log-odds {terms.attrs['raw_log_odds']:.4f}. Raw probability {terms.attrs['raw_probability']:.1%}.")
        plot(horizontal_effects(terms, "contribution", "Contribution to raw logistic log-odds (top 10)"), "lr_terms")
        st.dataframe(terms, hide_index=True)
        st.warning("The top 10 bars omit smaller terms. Only the complete table plus intercept reconstructs raw log-odds; these terms do not sum to the calibrated probability.")


def data_and_limits(bundle: ModelBundle) -> None:
    section("Data, decisions & limitations", "Understand the scope before interpreting any model output.")
    left, right = st.columns([1.3, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("**The source**")
        st.write("UCI Default of Credit Card Clients contains 30,000 records from Taiwan, with repayment, statement, and payment histories from April–September 2005. The binary target is default payment in the following month (1 = yes). Amounts are in New Taiwan dollars.")
        st.markdown("[UCI dataset and attribution](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [Dataset DOI](https://doi.org/10.24432/C55S3H)")
        st.caption("Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. CC BY 4.0. This project removes five source columns and renames the target.")
        st.markdown("**Financial-only preprocessing**")
        st.write("19 original financial inputs feed deterministic ratio and history features. Repayment status codes are one-hot encoded. Numeric medians and scaling are fitted on the training split only. Negative statement credits and unusual status codes are preserved.")
        st.write("Both models use fixed settings. A sigmoid calibrator is fitted on a separate split; no parameters are chosen using final test performance.")
        st.markdown("**Quality snapshot**")
        metric_row([
            ("Records", f"{len(bundle.dataset):,}", "All bundled financial records."),
            ("Missing cells", f"{bundle.dataset.isna().sum().sum():,}", "Measured in the bundled data; a training-only imputer still protects the pipeline."),
        ])
        duplicates = int(bundle.dataset.duplicated().sum())
        st.caption(f"{duplicates:,} identical financial-feature + target rows after excluded columns are removed. They are retained; random splitting may place identical histories across splits, making evaluation optimistic.")
    with right, st.container(border=True):
        st.markdown("**What this project cannot establish**")
        for text in [
            "Performance on a 2005 Taiwan credit card dataset does not establish performance for current borrowers, other markets, or loan origination.",
            "Removing demographic columns does not remove proxy bias or establish fairness. This project does not perform a demographic fairness audit.",
            "A random split is not an out-of-time validation. Repeated financial histories may cross splits; drift and generalization remain uncertain.",
            "Calibration is a group-level diagnostic. It does not guarantee an individual's probability or future calibration.",
            "Local substitutions and coefficients describe model behavior, not causes or approved explanations for credit decisions.",
            "No income, affordability, exposure-at-default, or loss-given-default model is included. Probability of default is not expected monetary loss.",
        ]:
            st.markdown(f"• {text}")
        st.warning("Educational portfolio project only. Do not use for credit approval, pricing, collections, regulatory reporting, or decisions about real people.")
    with st.expander("Financial input dictionary"):
        st.dataframe(pd.DataFrame([
            {"Field": "LIMIT_BAL", "Meaning": "Given credit limit", "Unit / encoding": "NT$"},
            {"Field": "PAY_0, PAY_2 … PAY_6", "Meaning": "September to April repayment status", "Unit / encoding": "Categorical codes −2 to 9"},
            {"Field": "BILL_AMT1 … BILL_AMT6", "Meaning": "September to April statement amounts", "Unit / encoding": "NT$; negatives are credit balances"},
            {"Field": "PAY_AMT1 … PAY_AMT6", "Meaning": "September to April previous payment amounts", "Unit / encoding": "NT$; nonnegative"},
            {"Field": "default", "Meaning": "Observed default payment next month", "Unit / encoding": "Target: 1 = yes, 0 = no"},
        ]), hide_index=True, width="stretch")


def main() -> None:
    with st.sidebar:
        st.title("LendLens ◈")
        st.caption("C R E D I T   R I S K   L A B")
        page = st.radio("Explore the lab", ["Portfolio overview", "Model laboratory", "Scenario studio", "Data & limitations"], label_visibility="collapsed")
        name = st.selectbox("Model", MODEL_NAMES, index=1, help="Both models share the same splits and excluded attributes.")
        threshold = st.slider("Teaching threshold", .05, .95, .50, .05, format="%.2f",
                              help="A hypothetical probability cut-off for diagnostics. This is not credit policy.")
        st.caption("All displayed model probabilities are sigmoid calibrated unless explicitly labelled raw.")
        st.markdown("**EDUCATIONAL ONLY**")
        st.caption("Built for learning, not real credit decisions. Use fictional scenario inputs only.")
        st.caption("UCI 350 · Taiwan, 2005\n\n19 financial inputs · 2 models\n\n60 / 20 / 20 experiment")
    st.caption("LENDLENS  /  FINTECH PORTFOLIO  /  DEFAULT PREDICTION")
    try:
        digest = hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
        with st.spinner("Preparing the reproducible experiment · first start fits and calibrates both models…"):
            bundle = fitted_experiment(digest)
        probabilities = test_probabilities(digest, bundle)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(f"The experiment could not be prepared: {exc}")
        st.code("python scripts/download_data.py\npython -m streamlit run app.py")
        st.stop()
    if page == "Portfolio overview":
        overview(bundle, probabilities, name, threshold)
    elif page == "Model laboratory":
        model_lab(bundle, probabilities, name, threshold, digest)
    elif page == "Scenario studio":
        scenario_studio(bundle, name, threshold)
    else:
        data_and_limits(bundle)
    st.caption("LendLens · A Year 2 FinTech learning project · UCI Default of Credit Card Clients. Historical credit card default is the study target. This system is not suitable for real credit approval.")


if __name__ == "__main__":
    main()
