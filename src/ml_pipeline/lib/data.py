"""Datasets for the churn pipeline.

CI never touches the network. The default dataset is a deterministic synthetic
churn frame built from ``make_classification`` (the signal) plus a few hand-
assembled categorical/numeric columns so the feature pipeline has a realistic
mix to transform. Point ``dataset`` at a CSV (with a ``target`` column name) to
run on real data instead.

The same generator, called with a positive ``shift``, produces a *shifted*
"current" frame for the drift command — a defensible, reproducible drift signal
rather than a hand-waved one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ml_pipeline.output import CliError

# Column groups the feature pipeline keys off. Engineered on top of the raw
# make_classification informative features so the frame reads like a real
# churn table (tenure, monthly spend, support calls, contract type, ...).
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "logins_last_30d",
]
CATEGORICAL_FEATURES = ["contract", "payment_method", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_CONTRACTS = ["month_to_month", "one_year", "two_year"]
_PAYMENTS = ["card", "bank_transfer", "ideal", "invoice"]
_REGIONS = ["noord", "oost", "zuid", "west", "randstad"]


def synthetic_churn(
    n_samples: int = 4000,
    seed: int = 42,
    shift: float = 0.0,
) -> pd.DataFrame:
    """Deterministic synthetic churn frame.

    ``shift`` (>= 0) pushes the distribution of a few features so the drift
    command has a clearly-shifted "current" set. The mapping from latent signal
    to columns is fixed, so the same ``(n_samples, seed, shift)`` always yields
    the same frame.

    NOTE: for *drift*, do not compare two ``synthetic_churn`` frames generated
    with different seeds — ``make_classification`` reseeds its latent cluster
    structure per seed, so different seeds are different *distributions*, not
    different samples of one distribution, and PSI would (correctly) flag them
    even at ``shift=0``. Use :func:`reference_current_split`, which draws one
    structure and partitions it, so ``shift=0`` means genuinely no drift.
    """
    rng = np.random.default_rng(seed)
    x, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.73, 0.27],  # realistic ~27% churn base rate
        class_sep=1.1,
        flip_y=0.02,
        random_state=seed,
    )
    return _assemble(x, y, rng, shift)


def reference_current_split(
    n_samples: int = 4000,
    seed: int = 42,
    shift: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One distribution, split into a reference and a current partition.

    Draws ``2 * n_samples`` rows from a single ``make_classification`` structure,
    shuffles, and splits in half: the first half is the reference, the second is
    the current set. Only the current set has ``shift`` applied. At ``shift=0``
    the two halves are genuinely the same distribution (PSI ~= 0); at
    ``shift>0`` the current set drifts on the shifted features. This is the
    honest reference-vs-current setup a drift monitor sees in production.
    """
    rng = np.random.default_rng(seed)
    x, y = make_classification(
        n_samples=2 * n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.73, 0.27],
        class_sep=1.1,
        flip_y=0.02,
        random_state=seed,
    )
    order = rng.permutation(2 * n_samples)
    x, y = x[order], y[order]
    ref = _assemble(x[:n_samples], y[:n_samples], np.random.default_rng(seed + 1), shift=0.0)
    cur = _assemble(x[n_samples:], y[n_samples:], np.random.default_rng(seed + 2), shift=shift)
    return ref, cur


def _assemble(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, shift: float) -> pd.DataFrame:
    """Map latent signal columns onto realistic churn columns, with optional shift."""
    n = len(y)
    # Numeric features: standardized signal mapped onto sensible scales, then
    # the optional shift so "current" data drifts vs "reference".
    tenure = np.clip(36 + 12 * x[:, 0] + shift * 18, 0, 84).round(1)
    monthly = np.clip(65 + 18 * x[:, 1] + shift * 25, 15, 140).round(2)
    total = (tenure * monthly + rng.normal(0, 50, n)).clip(min=0).round(2)
    support = np.clip(np.rint(2 + 1.5 * x[:, 2] + shift * 3), 0, 20).astype(int)
    logins = np.clip(np.rint(20 + 8 * x[:, 3] - shift * 6), 0, 90).astype(int)

    # Categorical features: contract from the churn signal (month-to-month
    # churns more); shift the payment-method mix under drift.
    contract_idx = np.clip(np.rint(1 - 0.9 * x[:, 4]).astype(int), 0, 2)
    contract = np.array(_CONTRACTS)[contract_idx]

    pay_p = np.array([0.4, 0.25, 0.25, 0.1])
    if shift:
        pay_p = pay_p + np.array([0.3, -0.1, -0.1, -0.1]) * min(shift, 1.0)
        pay_p = np.clip(pay_p, 0.01, None)
        pay_p = pay_p / pay_p.sum()
    payment = rng.choice(_PAYMENTS, size=n, p=pay_p)
    region = rng.choice(_REGIONS, size=n)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": support,
            "logins_last_30d": logins,
            "contract": contract,
            "payment_method": payment,
            "region": region,
            "churn": y.astype(int),
        }
    )


def load(dataset: str, target: str, n_samples: int, seed: int) -> pd.DataFrame:
    """Resolve a config ``dataset`` to a frame with a ``churn`` target column.

    ``synthetic`` uses the deterministic generator; anything else is treated as
    a path to a CSV whose ``target`` column is renamed to ``churn``.
    """
    if dataset == "synthetic":
        return synthetic_churn(n_samples=n_samples, seed=seed)

    p = Path(dataset)
    if not p.is_file():
        raise CliError(f"dataset not found: {p} (use 'synthetic' or a CSV path)")
    df = pd.read_csv(p)
    if target not in df.columns:
        raise CliError(f"target column '{target}' not in {p} (columns: {list(df.columns)})")
    df = df.rename(columns={target: "churn"})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise CliError(f"CSV {p} is missing expected feature columns: {missing}")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df["churn"].astype(int)
