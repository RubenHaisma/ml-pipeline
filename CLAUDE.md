# ml-pipeline — agent rules

For Claude Code and similar agents. Humans read `README.md`.

## What this repo is
Classic ML done right: a CLI-first (`mlp`) churn-style tabular classification
pipeline. It derives from [`ml-pipeline-template`](https://github.com/rubenhaisma/ml-pipeline-template)
and keeps that operational shell (CLI, MLflow, eval-with-baseline, `--json`),
but the *ML rigor* is the point here — an honest baseline, probability
**calibration** (Brier before/after), and **drift monitoring** (PSI + KS per
feature). Everything runs on CPU, deterministically, with no network downloads.

## Hard rules
1. **CLI-first.** Every capability ships as an `mlp` subcommand. No notebook-
   only or UI-only flows.
2. **`--json` on every command.** Machine-readable output is a contract, not a
   nicety. Don't add a command without it.
3. **Exit codes mean something.** `0` success; non-zero failure with one
   human-readable line on stderr (see `output.py`). Never swallow errors.
4. **MLflow is the single source of truth** for runs/params/metrics. No state
   in ad-hoc dotfiles. Falls back to `sqlite:///mlflow.db` with no server.
5. **Report a baseline with every metric.** A model that doesn't beat its
   `DummyClassifier` baseline is a finding to surface, not a number to hide.
6. **Calibration and drift are honest.** Report Brier before/after even when it
   doesn't improve. Flag drifted features by PSI threshold, don't hand-wave.
7. **Deterministic + offline.** The synthetic generator is seeded; CI never
   downloads. A real CSV is opt-in via the config `dataset`/`target` keys.
8. **Marimo `.py`, never `.ipynb`.** JSON notebooks don't diff, grep, or edit.

## Layout
```
src/ml_pipeline/
  cli.py            # Typer app, one command per capability
  output.py         # emit()/fail() — the output + exit-code contract
  commands/         # one file per subcommand (doctor/train/eval/calibrate/drift/infer)
  lib/              # config, tracking, data, model, calibration, drift (in-process)
notebooks/          # marimo .py (reliability diagram, drift report)
configs/            # yaml, one per run shape
tests/              # pytest smoke — runs the real train->...->infer loop on CPU
```

## What NOT to add
- `.ipynb` notebooks.
- Network downloads in the default path — the synthetic generator is the CI dataset.
- Cloud-specific SDKs (SageMaker/Vertex/Azure ML) — stays local + compose.
- A web dashboard — use MLflow's UI.
- Commands without `--json`.
