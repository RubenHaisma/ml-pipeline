"""``mlp drift`` — PSI + KS per feature between a reference and a shifted set."""

from __future__ import annotations

import typer

from ml_pipeline.lib import data, tracking
from ml_pipeline.lib import drift as drift_lib
from ml_pipeline.lib.config import TrainConfig
from ml_pipeline.lib.drift import PSI_THRESHOLD
from ml_pipeline.output import CliError, emit, fail


def drift(
    reference: str = typer.Option(
        ..., "--reference", help="train config defining the reference set"
    ),
    shift: float = typer.Option(
        0.5, "--shift", help="distribution shift applied to the synthetic current set"
    ),
    threshold: float = typer.Option(
        PSI_THRESHOLD, "--threshold", help="PSI threshold to flag a feature as drifted"
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        cfg = TrainConfig.from_yaml(reference)
        if cfg.dataset != "synthetic":
            raise CliError(
                "drift --shift only synthesizes a current set for the 'synthetic' dataset; "
                "for a real dataset, point reference/current at two CSVs in code"
            )
        ref_df, cur_df = data.reference_current_split(
            n_samples=cfg.n_samples, seed=cfg.seed, shift=shift
        )
        result = drift_lib.feature_drift(ref_df, cur_df, threshold=threshold)
        result["shift"] = shift

        with tracking.run(experiment=cfg.name, run_name=f"drift-shift{shift}"):
            tracking.log_metrics(
                {"n_drifted": result["n_drifted"], "shift": shift}
                | {f"psi_{f['feature']}": f["psi"] for f in result["features"]}
            )
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    flagged = result["drifted_features"]
    head = f"[red]{result['n_drifted']} drifted[/red]" if flagged else "[green]no drift[/green]"
    rows = "\n".join(
        f"  {'[red]!' if f['drifted'] else '[green] '}[/] {f['feature']:<16} psi {f['psi']}"
        for f in result["features"]
    )
    human = f"{head} at PSI>={threshold} (shift {shift})\n  flagged: {flagged or 'none'}\n{rows}"
    emit({"ok": True, **result}, json_out=json_out, human=human)
