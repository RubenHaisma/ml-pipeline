"""Feature drift report — PSI + KS between a reference and a shifted current set.

Marimo, not Jupyter. Computes the same PSI/KS drift report the `mlp drift`
command logs to MLflow, and ranks features by PSI so you can see *what* moved.

Run with: marimo edit notebooks/02_drift.py
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Feature drift report
        PSI (Population Stability Index) and the KS two-sample test per feature
        between the reference distribution (what the model trained on) and a
        synthetically shifted "current" distribution. Conventional PSI reading:
        `< 0.1` stable, `0.1–0.25` moderate, `>= 0.25` significant drift.
        Move the slider to see how much shift it takes to flag each feature.
        """
    )
    return


@app.cell
def __(mo):
    shift = mo.ui.slider(0.0, 1.0, value=0.5, step=0.1, label="distribution shift")
    shift
    return (shift,)


@app.cell
def __(shift):
    from ml_pipeline.lib import data
    from ml_pipeline.lib import drift as drift_lib
    from ml_pipeline.lib.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/churn.yaml")
    ref, cur = data.reference_current_split(
        n_samples=cfg.n_samples, seed=cfg.seed, shift=shift.value
    )
    report = drift_lib.feature_drift(ref, cur)
    return cfg, cur, data, drift_lib, ref, report


@app.cell
def __(mo, report):
    rows = "\n".join(
        f"| {f['feature']} | {f['type']} | {f['psi']} | "
        f"{'YES' if f['drifted'] else 'no'} | {f['ks_pvalue']} |"
        for f in report["features"]
    )
    mo.md(
        f"**Flagged (PSI >= {report['threshold']}):** "
        f"`{report['drifted_features'] or 'none'}`\n\n"
        "| feature | type | PSI | drifted | KS p-value |\n"
        "|---|---|---|---|---|\n" + rows
    )
    return


if __name__ == "__main__":
    app.run()
