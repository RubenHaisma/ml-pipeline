"""Drift monitoring — PSI + KS per feature, the production-ops half of ML.

A model that was honest at training time silently rots when the world shifts.
This module quantifies that shift between a *reference* set (what the model was
trained on) and a *current* set:

- **PSI (Population Stability Index)** — the industry-standard binned divergence
  measure. The conventional reading: < 0.1 stable, 0.1-0.25 moderate shift,
  >= 0.25 significant shift. We flag at 0.25 by default.
- **KS test** — a distribution-free two-sample test per numeric feature; a small
  p-value is statistical evidence the two samples differ.

Both are real, defensible, and computed per feature so you can see *what* moved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ml_pipeline.lib import data

PSI_THRESHOLD = 0.25
_EPS = 1e-6


def _psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI for a numeric feature, binned on reference quantiles."""
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if quantiles.size < 2:  # constant reference feature
        return 0.0
    edges = quantiles.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """PSI for a categorical feature, over the union of observed categories."""
    cats = pd.Index(ref.unique()).union(cur.unique())
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = PSI_THRESHOLD,
) -> dict[str, Any]:
    """Per-feature PSI (+ KS for numerics) between reference and current frames."""
    per_feature: list[dict[str, Any]] = []
    for feat in data.FEATURES:
        is_numeric = feat in data.NUMERIC_FEATURES
        if is_numeric:
            ref_v = reference[feat].to_numpy(dtype=float)
            cur_v = current[feat].to_numpy(dtype=float)
            psi = _psi_numeric(ref_v, cur_v)
            ks_stat, ks_p = ks_2samp(ref_v, cur_v)
            ks_entry: dict[str, Any] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 6),
            }
        else:
            psi = _psi_categorical(reference[feat], current[feat])
            ks_entry = {"ks_statistic": None, "ks_pvalue": None}

        per_feature.append(
            {
                "feature": feat,
                "type": "numeric" if is_numeric else "categorical",
                "psi": round(psi, 4),
                "drifted": psi >= threshold,
                **ks_entry,
            }
        )

    drifted = [f["feature"] for f in per_feature if f["drifted"]]
    return {
        "threshold": threshold,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "features": sorted(per_feature, key=lambda f: f["psi"], reverse=True),
    }
