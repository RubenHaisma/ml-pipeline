"""ml_pipeline — classic ML done right, CLI-first and fully CPU-verified.

A churn-style tabular classification pipeline whose value is the rigor most
portfolios skip: an honest baseline, probability calibration (Brier before/
after), and drift monitoring (PSI + KS per feature). Built on the house style:
CLI-first, ``--json`` on every command, load-bearing exit codes, MLflow as the
single source of truth, marimo (not Jupyter) for exploration.
"""

__version__ = "0.1.0"
