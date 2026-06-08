"""``mlp calibrate <name>`` — calibrate probabilities, report Brier before/after."""

from __future__ import annotations

from pathlib import Path

import typer

from ml_pipeline.lib import calibration, tracking
from ml_pipeline.output import CliError, emit, fail


def calibrate(
    name: str = typer.Argument(..., help="project name (artifacts/<name>/)"),
    method: str = typer.Option("isotonic", "--method", help="isotonic | sigmoid"),
    out: str = typer.Option("artifacts", "--out", help="artifacts root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        out_dir = Path(out) / name
        result = calibration.calibrate(out_dir, method=method)
        with tracking.run(experiment=name, run_name=f"calibrate-{method}"):
            tracking.log_metrics(
                {
                    "brier_before": result["brier_before"],
                    "brier_after": result["brier_after"],
                    "brier_improvement": result["brier_improvement"],
                }
            )
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    verdict = "[green]improved[/green]" if result["improved"] else "[yellow]no improvement[/yellow]"
    human = (
        f"[green]calibrated[/green] {name} ({result['method']})\n"
        f"  brier {result['brier_before']} -> {result['brier_after']} "
        f"({result['brier_improvement']:+}) {verdict}\n"
        f"  calibrated model -> {result['calibrated_model_path']}"
    )
    emit({"ok": True, **result}, json_out=json_out, human=human)
