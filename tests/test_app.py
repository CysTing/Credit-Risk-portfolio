"""Exercise actual model training and user flows with Streamlit's test harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.scenarios import PRESETS


@pytest.mark.integration
def test_dashboard_pages_and_fictional_submission():
    path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(path), default_timeout=180).run()
    assert not app.exception
    assert not app.error
    assert any(metric.label == "ROC AUC" for metric in app.metric)

    # Both model families must remain usable; selection is explicit, not AUC-based.
    app.sidebar.selectbox[0].set_value("Logistic regression").run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Model laboratory").run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) >= 1

    app.sidebar.radio[0].set_value("Scenario studio").run()
    assert not app.exception
    presets = next(widget for widget in app.selectbox if widget.label == "Start from a fictional profile")
    presets.set_value(PRESETS[1]).run()
    assert not app.exception
    app.number_input[0].set_value(150_000)
    submit = next(button for button in app.button if button.label == "Explore this fictional scenario")
    submit.click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["fictional_profile"]["LIMIT_BAL"].iloc[0] == 150_000

    app.sidebar.selectbox[0].set_value("Random forest").run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Data & limitations").run()
    assert not app.exception
    assert not app.error
    assert any("What this project cannot establish" in item.value for item in app.markdown)
