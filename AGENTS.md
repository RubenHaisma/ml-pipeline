# AGENTS.md

Agent instructions for **ml-pipeline**, in the cross-tool [AGENTS.md](https://agents.md) format — read natively by Codex, Cursor, GitHub Copilot, Windsurf, Amp, Devin, and others. Claude Code reads `CLAUDE.md`, which is a **symlink to this file**, so there is a single source of truth for every tool. Humans: see [README.md](README.md).

## What this repo is

A CLI-first (`mlp`) **churn-style tabular classification** pipeline — classic ML done right. It derives from [`ml-pipeline-template`](https://github.com/RubenHaisma/ml-pipeline-template) and keeps that operational shell (CLI, MLflow, eval-with-baseline, `--json`), but here the **ML rigor is the point**:

- an **honest baseline** (`DummyClassifier`) reported with every metric, as `lift_over_baseline`;
- probability **calibration** (Brier before/after, isotonic or sigmoid);
- **drift monitoring** (PSI + KS per feature, with a flag threshold).

Everything runs on **CPU, deterministically, with no network downloads** — the dataset is a seeded synthetic generator; a real CSV is opt-in.

## Driving the CLI as an agent

Every command is **non-interactive**, takes **`--json`**, and uses **load-bearing exit codes** — so any agent (or script) can drive the whole loop and parse results with no TTY, no UI, no service. The example outputs below are real (seed 42):

```bash
mlp doctor --json
# -> {"ok": true, "checks": {"python": "3.12.9", "python_ok": true, "deps_missing": [],
#     "deps_ok": true, "docker": true, "tracking_uri": "sqlite:///mlflow.db"}}   exit 0 when ready

mlp train configs/churn.yaml --json
# -> {"ok": true, "name": "churn-gb", "model": "gradient_boosting",
#     "model_path": "artifacts/churn-gb/model.joblib",
#     "metrics": {"roc_auc": 0.863, "pr_auc": 0.7298, "f1": 0.664, "brier": 0.1247,
#                 "baseline_roc_auc": 0.5, "baseline_brier": 0.1994,
#                 "lift_over_baseline": 0.363, "positive_rate": 0.275},
#     "n_train": 3000, "n_test": 1000}

mlp eval churn-gb --json
# -> {"ok": true, "metrics": {"roc_auc": 0.863, ..., "lift_over_baseline": 0.363}, "n_test": 1000}

mlp calibrate churn-gb --json        # --method isotonic|sigmoid
# -> {"ok": true, "method": "isotonic", "calibrated_model_path": "...model_calibrated.joblib",
#     "brier_before": 0.1247, "brier_after": 0.1223, "brier_improvement": 0.0025,
#     "improved": true, "reliability_before": [...], "reliability_after": [...]}

mlp drift --reference configs/churn.yaml --json     # --shift, --threshold
# -> {"ok": true, "threshold": 0.25, "n_reference": 4000, "n_current": 4000,
#     "drifted_features": ["total_charges", "support_calls"], "n_drifted": 2,
#     "features": [{"feature": "support_calls", "type": "numeric", "psi": 0.4532,
#                   "drifted": true, "ks_statistic": 0.2895, "ks_pvalue": 0.0}, ...], "shift": 0.5}

mlp infer churn-gb --json-input '{"tenure_months": 3, "monthly_charges": 95.0, ...}' --json
# -> {"ok": true, "features": ["tenure_months", ...], "churn_probability": 0.0606, "churn_label": 0}
```

**Contract:** with `--json`, stdout is exactly one JSON object (success *or* failure: `{"ok": false, "error": "..."}`). Exit `0` = success, non-zero = failure with one line on stderr. So: parse stdout, branch on the exit code. Discover the surface with `mlp --help` and `mlp <cmd> --help`.

## Adapt it to your own data

The default `dataset: synthetic` is a seeded generator so CI is offline and deterministic. To point it at a real CSV, edit the config — copy `configs/churn.yaml` per experiment, one config = one run shape:

```yaml
name: my-churn
dataset: data/customers.csv   # path to a CSV instead of 'synthetic'
target: churned               # the label column in that CSV
model: gradient_boosting      # gradient_boosting | logistic
params: { n_estimators: 200, max_depth: 3, learning_rate: 0.1 }
test_size: 0.25
seed: 42
```

Then `mlp train configs/my-churn.yaml --json`. The feature pipeline (numeric impute/scale + categorical one-hot, via a `ColumnTransformer` + `Pipeline`) lives in `src/ml_pipeline/lib/data.py` and `lib/model.py`; calibration in `lib/calibration.py`; drift in `lib/drift.py`. Calibration and drift then run against `my-churn` the same way.

## Setup (for the agent's environment)

```bash
uv sync --extra dev      # install (CPU, fast — no GPU path exists)
uv run mlp doctor --json # confirm the environment is ready
```

## Hard rules (when editing this repo)

1. **CLI-first.** Every capability ships as an `mlp` subcommand. No notebook-only or UI-only flows.
2. **`--json` on every command.** Machine-readable output is a contract, not a nicety. Don't add a command without it.
3. **Exit codes mean something.** `0` success; non-zero failure with one human-readable line on stderr (see `src/ml_pipeline/output.py`). Never swallow errors.
4. **MLflow is the single source of truth** for runs/params/metrics. No state in ad-hoc dotfiles. Falls back to `sqlite:///mlflow.db` with no server.
5. **Report a baseline with every metric.** A model that doesn't beat its `DummyClassifier` baseline is a finding to surface, not a number to hide.
6. **Calibration and drift are honest.** Report Brier before/after even when it doesn't improve. Flag drifted features by PSI threshold, don't hand-wave.
7. **Deterministic + offline.** The synthetic generator is seeded; CI never downloads. A real CSV is opt-in via the config `dataset`/`target` keys.
8. **Marimo `.py`, never `.ipynb`.** JSON notebooks don't diff, grep, or edit surgically.

## Build / test / verify

```bash
uv run ruff check src tests scripts     # lint
uv run pytest                           # smoke suite — runs the real train->eval->calibrate->drift->infer loop on CPU
make repro                              # determinism: train twice, assert identical metrics
make readme                             # run the README's <!-- ci-test --> commands
```

## Layout

```
src/ml_pipeline/
  cli.py            # Typer app, one command per capability
  output.py         # emit()/fail() — the output + exit-code contract
  commands/         # one file per subcommand (doctor/train/eval/calibrate/drift/infer)
  lib/              # config, tracking, data, model, calibration, drift (in-process, no shellouts)
notebooks/          # marimo .py (reliability diagram, drift report)
configs/            # yaml, one per run shape (churn.yaml is the reference)
scripts/            # stdlib CI helpers (ci_report, test_readme, check_repro)
tests/              # pytest smoke — the real train->...->infer loop on CPU
```

## What NOT to add

- `.ipynb` notebooks. Network downloads in the default path (the synthetic generator is the CI dataset). Cloud-specific SDKs (SageMaker/Vertex/Azure ML) — stays local + compose. A web dashboard (use MLflow's UI). Commands without `--json`.
