"""``mlp eval <name>`` — recompute holdout metrics vs baseline from artifacts."""

from __future__ import annotations

from pathlib import Path

import typer

from ml_pipeline.lib import model, tracking
from ml_pipeline.output import CliError, emit, fail


def eval(
    name: str = typer.Argument(..., help="project name (artifacts/<name>/)"),
    out: str = typer.Option("artifacts", "--out", help="artifacts root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        out_dir = Path(out) / name
        result = model.reevaluate(out_dir)
        with tracking.run(experiment=name, run_name="eval"):
            tracking.log_metrics(result["metrics"])
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    m = result["metrics"]
    human = (
        f"[green]eval[/green] {name}  (n_test {result['n_test']})\n"
        f"  roc_auc {m['roc_auc']}  (baseline {m['baseline_roc_auc']}, "
        f"lift {m['lift_over_baseline']:+})\n"
        f"  pr_auc {m['pr_auc']}  f1 {m['f1']}  brier {m['brier']}"
    )
    emit({"ok": True, **result}, json_out=json_out, human=human)
