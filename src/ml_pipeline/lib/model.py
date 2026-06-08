"""The reference pipeline: a calibrated churn classifier on tabular data.

This is *classic ML done right*. A real sklearn ``ColumnTransformer`` (numeric
scaling + categorical one-hot) is wrapped in a ``Pipeline`` with the estimator,
the model is always scored against a ``DummyClassifier`` baseline, and the full
metric set (ROC-AUC, PR-AUC, F1, Brier, lift) is reported — a number without a
baseline is marketing, not evaluation.

Everything runs on CPU in a second or two so the happy path works on any laptop
and in CI with no GPU and no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.lib import data
from ml_pipeline.lib.config import TrainConfig
from ml_pipeline.output import CliError

_MODELS = {
    "gradient_boosting": GradientBoostingClassifier,
    "logistic": LogisticRegression,
}


def build_pipeline(model_name: str, params: dict[str, Any], seed: int = 42) -> Pipeline:
    """A real feature pipeline: scale numerics, one-hot categoricals, then fit.

    ``seed`` is threaded into the estimator's ``random_state`` so training is
    bit-for-bit reproducible: GradientBoosting (and LogisticRegression's solvers)
    carry internal randomness that, left unseeded, makes two runs of the same
    config disagree in the 4th decimal. An explicit caller ``random_state`` in
    ``params`` wins.
    """
    if model_name not in _MODELS:
        raise CliError(f"unknown model '{model_name}', choose from {sorted(_MODELS)}")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), data.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                data.CATEGORICAL_FEATURES,
            ),
        ]
    )
    estimator = _MODELS[model_name](random_state=seed, **params)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _proba(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    """Positive-class probability, robust across estimators."""
    return estimator.predict_proba(x)[:, 1]


def evaluate(
    estimator: Any,
    x_te: pd.DataFrame,
    y_te: pd.Series,
    baseline: Any,
) -> dict[str, float]:
    """Score an estimator against a baseline on a holdout. Lift is the headline."""
    proba = _proba(estimator, x_te)
    pred = (proba >= 0.5).astype(int)
    base_proba = baseline.predict_proba(x_te)[:, 1]

    roc = float(roc_auc_score(y_te, proba))
    base_roc = float(roc_auc_score(y_te, base_proba))
    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "brier": round(float(brier_score_loss(y_te, proba)), 4),
        "baseline_roc_auc": round(base_roc, 4),
        "baseline_brier": round(float(brier_score_loss(y_te, base_proba)), 4),
        "lift_over_baseline": round(roc - base_roc, 4),
        "positive_rate": round(float(y_te.mean()), 4),
    }


def train(cfg: TrainConfig, out_dir: Path) -> dict[str, Any]:
    """Train, evaluate against an honest baseline, persist the pipeline + split.

    The train/test split is saved next to the model so ``eval`` and
    ``calibrate`` recompute on the *same* holdout — reproducibility, not luck.
    """
    df = data.load(cfg.dataset, cfg.target, cfg.n_samples, cfg.seed)
    x, y = data.split_xy(df)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    pipe = build_pipeline(cfg.model, cfg.params, seed=cfg.seed)
    pipe.fit(x_tr, y_tr)

    baseline = DummyClassifier(strategy="prior").fit(x_tr, y_tr)
    metrics = evaluate(pipe, x_te, y_te, baseline)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path)
    # Persist the holdout so eval/calibrate are deterministic across invocations.
    joblib.dump(
        {"x_tr": x_tr, "x_te": x_te, "y_tr": y_tr, "y_te": y_te, "model": cfg.model},
        out_dir / "split.joblib",
    )

    return {
        "name": cfg.name,
        "model": cfg.model,
        "model_path": str(model_path),
        "metrics": metrics,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def _load_split(out_dir: Path) -> dict[str, Any]:
    split_path = out_dir / "split.joblib"
    if not split_path.is_file():
        raise CliError(f"split not found: {split_path} (train one first)")
    return joblib.load(split_path)


def reevaluate(out_dir: Path) -> dict[str, Any]:
    """Recompute holdout metrics vs baseline from the persisted model + split."""
    model_path = out_dir / "model.joblib"
    if not model_path.is_file():
        raise CliError(f"model not found: {model_path} (train one first)")
    pipe = joblib.load(model_path)
    s = _load_split(out_dir)
    baseline = DummyClassifier(strategy="prior").fit(s["x_tr"], s["y_tr"])
    metrics = evaluate(pipe, s["x_te"], s["y_te"], baseline)
    return {"metrics": metrics, "n_test": int(len(s["y_te"]))}


def predict_one(model_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Score a single example. ``row`` maps feature name -> value."""
    if not model_path.is_file():
        raise CliError(f"model not found: {model_path} (train one first)")
    pipe = joblib.load(model_path)
    missing = [c for c in data.FEATURES if c not in row]
    if missing:
        raise CliError(f"missing features: {missing} (need all of {data.FEATURES})")
    x = pd.DataFrame([{c: row[c] for c in data.FEATURES}])
    proba = float(_proba(pipe, x)[0])
    return {
        "churn_probability": round(proba, 4),
        "churn_label": int(proba >= 0.5),
    }
