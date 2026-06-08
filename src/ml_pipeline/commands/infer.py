"""``mlp infer`` — score one example against a trained model. The serving stand-in."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ml_pipeline.lib import data, model
from ml_pipeline.output import CliError, emit, fail

# A sensible default example so `mlp infer <name>` works with no extra flags.
_DEFAULT = {
    "tenure_months": 6,
    "monthly_charges": 95.0,
    "total_charges": 570.0,
    "support_calls": 5,
    "logins_last_30d": 4,
    "contract": "month_to_month",
    "payment_method": "card",
    "region": "randstad",
}


def infer(
    name: str = typer.Argument(..., help="project name (artifacts/<name>/model.joblib)"),
    json_input: str = typer.Option(
        None, "--json-input", help="one example as JSON, e.g. '{\"tenure_months\": 6, ...}'"
    ),
    out: str = typer.Option("artifacts", "--out", help="artifacts root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        if json_input is not None:
            try:
                row: dict[str, Any] = json.loads(json_input)
            except json.JSONDecodeError as exc:
                raise CliError(f"--json-input is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise CliError("--json-input must be a JSON object of feature -> value")
        else:
            row = dict(_DEFAULT)
        model_path = Path(out) / name / "model.joblib"
        result = model.predict_one(model_path, row)
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    verdict = "[red]churn[/red]" if result["churn_label"] else "[green]retain[/green]"
    human = f"{verdict}  p(churn)={result['churn_probability']}"
    emit({"ok": True, "features": data.FEATURES, **result}, json_out=json_out, human=human)
