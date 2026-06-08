# Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  mlp CLI (Typer)                                               │
│  doctor · train · eval · calibrate · drift · infer · version   │
└───────┬───────────────────────────────────────────┬──────────┘
        │                                             │
        ▼                                             ▼
  ┌──────────────────────────┐                 ┌───────────┐
  │ lib/                      │  ── trains ──►  │  MLflow   │
  │  data      (synthetic)    │                 │  runs +   │
  │  model     (Pipeline +    │  ── logs ─────► │  params + │
  │             baseline)     │                 │  metrics  │
  │  calibration (Brier)      │                 └───────────┘
  │  drift      (PSI + KS)    │
  └──────────┬───────────────┘
             │
             ▼
  artifacts/<name>/{model,split,model_calibrated}.joblib  ──►  mlp eval / calibrate / infer
```

## The pipeline (`lib/model.py`)
A real sklearn `ColumnTransformer` — `StandardScaler` on the numeric columns,
`OneHotEncoder(handle_unknown="ignore")` on the categoricals — wrapped in a
`Pipeline` with the estimator (`GradientBoostingClassifier` or
`LogisticRegression`). Training stratifies the split, fits the pipeline, fits a
`DummyClassifier(strategy="prior")` baseline, and reports ROC-AUC, PR-AUC, F1,
Brier, and **lift over baseline** together. The exact train/test split is saved
next to the model (`split.joblib`) so `eval` and `calibrate` recompute on the
*same* holdout — reproducibility, not luck.

## Calibration (`lib/calibration.py`)
`CalibratedClassifierCV` is fit with 5-fold cross-validation on the training
split (it never trains and calibrates on the same rows), then Brier is compared
**before vs after** on the untouched holdout, with per-bin reliability-curve
data. If calibration doesn't improve Brier, that is reported honestly — it's a
finding, not a failure to hide.

## Drift (`lib/drift.py`)
Per-feature **PSI** (binned on reference quantiles for numerics, over the
category union for categoricals) plus a **KS two-sample test** for numerics,
between a reference set and a (synthetically shifted) current set. Features at
or above the PSI threshold (default `0.25`) are flagged. This is the production-
ops half of ML: a model that was honest at training time rots silently when the
world shifts, and this quantifies exactly *what* moved.

## Dataset (`lib/data.py`)
A deterministic synthetic churn frame: `make_classification` provides the latent
signal (seeded, ~27% churn base rate), and a fixed mapping assembles realistic
numeric (tenure, monthly charges, support calls, ...) and categorical (contract,
payment method, region) columns on top. CI never downloads anything. Point the
config `dataset` at a CSV path (with a `target` column) to run on real data.

## Output contract (`output.py`)
Every command funnels through `emit()` (success) or `fail()` (error). This is
what makes `--json` and exit codes uniform instead of per-command guesswork:

- `--json` → exactly one JSON object on stdout, success or failure.
- human mode → rich-formatted line(s) on stdout, `error: ...` on stderr.
- failure → non-zero exit, always.

## Tracking (`lib/tracking.py`)
MLflow with a local `sqlite:///mlflow.db` fallback so a fresh checkout works
with no services (MLflow 3 deprecated the bare file store; sqlite is the
supported local backend). Export `MLFLOW_TRACKING_URI=http://localhost:5050`
(after `make up`) to use the shared backend in `docker-compose.yml`.
