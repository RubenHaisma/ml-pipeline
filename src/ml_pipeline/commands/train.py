"""``mlp train <config>`` — train, evaluate vs baseline, log everything to MLflow."""

from __future__ import annotations

from pathlib import Path

import typer

from ml_pipeline.lib import model, tracking
from ml_pipeline.lib.config import TrainConfig
from ml_pipeline.output import CliError, emit, fail


def train(
    config: str = typer.Argument(..., help="path to a train config yaml"),
    out: str = typer.Option("artifacts", "--out", help="where to write the model"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        cfg = TrainConfig.from_yaml(config)
        out_dir = Path(out) / cfg.name
        result = model.train(cfg, out_dir)

        with tracking.run(experiment=cfg.name, run_name=f"train-{cfg.model}"):
            tracking.log_params(
                {"model": cfg.model, "seed": cfg.seed, "n_samples": cfg.n_samples, **cfg.params}
            )
            tracking.log_metrics(result["metrics"])
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    m = result["metrics"]
    human = (
        f"[green]trained[/green] {cfg.name} ({cfg.model})\n"
        f"  roc_auc {m['roc_auc']}  (baseline {m['baseline_roc_auc']}, "
        f"lift {m['lift_over_baseline']:+})\n"
        f"  pr_auc {m['pr_auc']}  f1 {m['f1']}  brier {m['brier']}\n"
        f"  model -> {result['model_path']}"
    )
    emit({"ok": True, **result}, json_out=json_out, human=human)
