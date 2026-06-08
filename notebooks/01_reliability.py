"""Reliability diagram — does the model's 0.8 actually churn 0.8 of the time?

Marimo, not Jupyter: plain Python that diffs, greps, and edits like source.
Trains the reference config, calibrates it, and shows the reliability curve
before vs after calibration (the diagonal is perfect calibration).

Run with: marimo edit notebooks/01_reliability.py
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
        # Reliability (calibration) diagram
        A model can rank well (high ROC-AUC) yet output miscalibrated
        probabilities. We plot, per probability bin, the mean predicted
        probability vs the observed churn rate. Points on the diagonal are
        perfectly calibrated. Calibration should pull the curve toward it.
        """
    )
    return


@app.cell
def __():
    from pathlib import Path

    from ml_pipeline.lib import calibration, model
    from ml_pipeline.lib.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/churn.yaml")
    out_dir = Path("artifacts") / cfg.name
    model.train(cfg, out_dir)
    cal = calibration.calibrate(out_dir)
    return cal, cfg, out_dir


@app.cell
def __(cal, mo):
    before = cal["reliability_before"]
    after = cal["reliability_after"]
    mo.md(
        f"**Brier before:** `{cal['brier_before']}`  "
        f"**after:** `{cal['brier_after']}`  "
        f"**improvement:** `{cal['brier_improvement']:+}` "
        f"({'improved' if cal['improved'] else 'no improvement'})"
    )
    return after, before


@app.cell
def __(after, before, mo):
    def _curve(bins):
        return "\n".join(
            f"| {b['range']} | {b['mean_predicted']} | {b['observed']} | {b['count']} |"
            for b in bins
        )

    header = "| bin range | mean predicted | observed | count |\n|---|---|---|---|\n"
    mo.md(
        "### Before calibration\n"
        + header
        + _curve(before)
        + "\n\n### After calibration\n"
        + header
        + _curve(after)
    )
    return


if __name__ == "__main__":
    app.run()
