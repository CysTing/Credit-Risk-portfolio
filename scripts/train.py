"""Reproduce all final-test reports without persisting executable model files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn

from credit_risk.data import DEFAULT_DATA_PATH, load_dataset
from credit_risk.explain import permutation_importance_table
from credit_risk.modeling import (
    MODEL_NAMES, SEED, evaluate_models, financial_history_overlap, reliability_table,
    train_models, training_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--importance", action="store_true", help="Also compute held-out permutation importance.")
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    bundle = train_models(load_dataset(args.data), n_estimators=args.trees)
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = evaluate_models(bundle)
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    diagnostics = training_diagnostics(bundle)
    diagnostics.to_csv(args.output / "train_test_comparison.csv", index=False)
    s = bundle.splits
    split_report = {}
    for name, X, y in [("train", s.X_train, s.y_train), ("calibration", s.X_cal, s.y_cal), ("test", s.X_test, s.y_test)]:
        split_report[name] = {
            "rows": len(y), "positive_count": int(y.sum()), "positive_rate": float(y.mean()),
            "index_sha256": hashlib.sha256(
                ",".join(map(str, sorted(X.index))).encode("utf-8")
            ).hexdigest(),
        }
    overlap = financial_history_overlap(s)
    metadata = {
        "seed": SEED, "split": "stratified 60/20/20", "threshold": .5,
        "calibration": "Frozen fitted training pipeline; sigmoid fitted only on calibration split",
        "model_selection": "Both fixed model families reported. No test-selected model or threshold.",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "hyperparameters": {
            "logistic": {"C": 1., "max_iter": 1500, "solver": "lbfgs"},
            "forest": {"n_estimators": args.trees, "max_depth": 10, "min_samples_leaf": 15, "max_features": .7},
        },
        "splits": split_report, "financial_history_overlap": overlap,
        "training_diagnostics": diagnostics.to_dict(orient="records"),
        "evaluation": comparison.to_dict(orient="records"),
    }
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "splits.json").write_text(
        json.dumps({"splits": split_report, "financial_history_overlap": overlap}, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        for version, estimator in [("raw", bundle.models[name]), ("calibrated", bundle.calibrated[name])]:
            prob = estimator.predict_proba(s.X_test)[:, 1]
            reliability_table(s.y_test, prob).to_csv(args.output / f"reliability_{slug}_{version}.csv", index=False)
        if args.importance:
            importance = permutation_importance_table(bundle, name)
            importance.to_csv(args.output / f"feature_importance_{slug}.csv", index=False)
    print(comparison[["model", "version", "roc_auc", "average_precision", "brier_score", "recall"]].to_string(index=False))
    print(f"Reports saved to {args.output}")


if __name__ == "__main__":
    main()
