"""Probability calibration — the step most churn demos skip.

A model that ranks well (high ROC-AUC) can still output badly miscalibrated
probabilities: when it says 0.8 it might actually churn 0.5 of the time. For a
churn model whose probabilities feed a retention budget, that gap is the whole
ballgame. We fit a ``CalibratedClassifierCV`` and report the **Brier score**
before and after, plus reliability-curve bins, so the improvement (or lack of
it — also a finding) is on the table, not assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from ml_pipeline.lib import model as model_lib
from ml_pipeline.output import CliError


def reliability_bins(
    y_true: pd.Series, proba: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Reliability-curve data: per-bin mean predicted prob vs observed rate.

    A perfectly calibrated model has ``mean_predicted`` ~= ``observed`` in every
    populated bin. These are the points a reliability diagram plots.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    y = np.asarray(y_true)
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            {
                "bin": b,
                "range": [round(float(edges[b]), 2), round(float(edges[b + 1]), 2)],
                "count": count,
                "mean_predicted": round(float(proba[mask].mean()), 4),
                "observed": round(float(y[mask].mean()), 4),
            }
        )
    return bins


def calibrate(out_dir: Path, method: str = "isotonic") -> dict[str, Any]:
    """Calibrate the trained pipeline on the persisted holdout.

    Calibration is fit with cross-validation on the *training* split (so it
    never sees the test holdout), then Brier and reliability are reported on the
    untouched holdout — a clean before/after comparison.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise CliError(f"unknown calibration method '{method}' (isotonic | sigmoid)")

    model_path = out_dir / "model.joblib"
    if not model_path.is_file():
        raise CliError(f"model not found: {model_path} (train one first)")
    base_pipe = joblib.load(model_path)
    s = model_lib._load_split(out_dir)
    x_tr, y_tr, x_te, y_te = s["x_tr"], s["y_tr"], s["x_te"], s["y_te"]

    proba_before = base_pipe.predict_proba(x_te)[:, 1]
    brier_before = float(brier_score_loss(y_te, proba_before))

    # Refit a fresh pipeline inside CalibratedClassifierCV via cross-val so the
    # calibrator is honest (it never trains and calibrates on the same rows).
    fresh = model_lib.build_pipeline(s["model"], {})
    fresh.set_params(**{k: v for k, v in base_pipe.get_params().items() if k in fresh.get_params()})
    calibrated = CalibratedClassifierCV(fresh, method=method, cv=5)
    calibrated.fit(x_tr, y_tr)

    proba_after = calibrated.predict_proba(x_te)[:, 1]
    brier_after = float(brier_score_loss(y_te, proba_after))

    cal_path = out_dir / "model_calibrated.joblib"
    joblib.dump(calibrated, cal_path)

    return {
        "method": method,
        "calibrated_model_path": str(cal_path),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "brier_improvement": round(brier_before - brier_after, 4),
        "improved": brier_after <= brier_before,
        "reliability_before": reliability_bins(y_te, proba_before),
        "reliability_after": reliability_bins(y_te, proba_after),
    }
