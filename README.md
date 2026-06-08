# ml-pipeline

**Classic ML done right.** A CLI-first churn-style tabular classification pipeline whose value is the rigor most portfolios skip: an **honest baseline**, probability **calibration**, and **drift monitoring**. Train → eval-against-baseline → calibrate → monitor drift → serve, tracked in MLflow, driven by one machine-readable binary (`mlp`). Fully CPU-runnable, deterministic, and verified end-to-end.

> Derives from [`ml-pipeline-template`](https://github.com/rubenhaisma/ml-pipeline-template) — same operational shell (CLI, `--json`, load-bearing exit codes, MLflow as source of truth). The difference here is the ML: not a model that ranks well in a notebook, but one with a baseline you can trust, probabilities you can calibrate, and a drift report you can act on.

## Why it looks like this

Most churn demos report an accuracy number and stop. The three things that separate that from a model you'd put in front of a retention budget are exactly the three things this repo does:

- **Honest baseline** — every metric is reported next to a `DummyClassifier`. A model that doesn't beat its baseline is a finding, not a number to hide. ROC-AUC, PR-AUC, F1, Brier, and **lift over baseline** are reported together.
- **Calibration** — a model can rank well (high ROC-AUC) and still output badly miscalibrated probabilities. `mlp calibrate` fits `CalibratedClassifierCV` and reports **Brier before/after** plus reliability-curve bins, so the improvement (or lack of it) is on the table.
- **Drift monitoring** — `mlp drift` computes **PSI** and a **KS test** per feature between a reference set and a (synthetically shifted) current set, and flags features over a PSI threshold. The production-ops half of ML, not an afterthought.

Plus the house style: **CLI-first**, **`--json` on every command**, **load-bearing exit codes**, **MLflow as the single source of truth**, and **marimo `.py` notebooks** (never `.ipynb`).

## Quickstart

```bash
uv sync --extra dev          # install (CPU only, no GPU, no downloads)
uv run mlp doctor            # environment readiness check (--json for CI)
uv run mlp train configs/churn.yaml
uv run mlp eval churn-gb
uv run mlp calibrate churn-gb
uv run mlp drift --reference configs/churn.yaml --shift 0.5
uv run mlp infer churn-gb
```

Output of `train` (human mode):

```
trained churn-gb (gradient_boosting)
  roc_auc 0.8628  (baseline 0.5, lift +0.3628)
  pr_auc 0.7295  f1 0.664  brier 0.1248
  model -> artifacts/churn-gb/model.joblib
```

The `DummyClassifier(strategy="prior")` baseline has no ranking power (ROC-AUC `0.5`), so the `+0.36` lift is real signal. `mlp calibrate` then improves Brier from `0.1248` to `0.1223`, and `mlp drift --shift 0.5` flags `support_calls` and `total_charges` (PSI `>= 0.25`) while leaving the unshifted features stable — see below.

`make demo` runs the full rigor loop (train → eval → calibrate → drift → infer).

## CLI surface

```
mlp doctor [--json]                                  # is this environment ready?
mlp train <config> [--out] [--json]                  # train, eval vs baseline, log to MLflow
mlp eval <name> [--out] [--json]                     # recompute holdout metrics vs baseline
mlp calibrate <name> [--method] [--out] [--json]     # calibrate, Brier before/after + reliability
mlp drift --reference <cfg> [--shift] [--threshold] [--json]   # PSI + KS per feature, flag drift
mlp infer <name> [--json-input] [--out] [--json]     # score one example -> churn probability
mlp version [--json]
```

Every command emits a single JSON object with `--json` and exits non-zero on failure with one line on stderr.

`infer` takes a full example as JSON:

```bash
uv run mlp infer churn-gb --json-input \
  '{"tenure_months": 2, "monthly_charges": 110.0, "total_charges": 220.0, "support_calls": 8, "logins_last_30d": 1, "contract": "month_to_month", "payment_method": "card", "region": "randstad"}'
```

## Dataset

The default dataset is a **deterministic synthetic churn frame** (`sklearn.make_classification` for the latent signal + hand-assembled numeric/categorical columns, fixed seed, ~27% churn base rate), so CI never downloads anything and results reproduce exactly. To run on **real data**, point the config at a CSV:

```yaml
# configs/churn.yaml
dataset: data/customers.csv   # CSV path instead of 'synthetic'
target: Churn                 # your label column; renamed to 'churn' internally
```

The CSV must contain the expected feature columns (`tenure_months`, `monthly_charges`, `total_charges`, `support_calls`, `logins_last_30d`, `contract`, `payment_method`, `region`) — adapt `src/ml_pipeline/lib/data.py` to your schema.

## Tracking & notebooks

```bash
make up                                   # MLflow on localhost:5050
export MLFLOW_TRACKING_URI=http://localhost:5050
uv run marimo edit notebooks/01_reliability.py   # reliability diagram before/after calibration
uv run marimo edit notebooks/02_drift.py         # PSI + KS feature drift report
```

Without a server the CLI falls back to a local `sqlite:///mlflow.db` store.

## What's verified

Everything in this repo is verified on CPU — no GPU path exists, nothing is hand-waved.

| Path                                            | Status      |
| ----------------------------------------------- | ----------- |
| `mlp train` / `eval` / `infer` on CPU           | ✅ verified |
| Feature pipeline (ColumnTransformer + Pipeline) | ✅ verified |
| Model beats `DummyClassifier` baseline on ROC-AUC | ✅ verified |
| `mlp calibrate` reduces/maintains Brier         | ✅ verified |
| `mlp drift` PSI + KS flags a shifted feature    | ✅ verified |
| `pytest` smoke + rigor suite + ruff in CI       | ✅ verified |
| MLflow local sqlite store                       | ✅ verified |
| MLflow server via docker-compose                | ✅ compose provided, runs locally |

## License

Apache-2.0.
