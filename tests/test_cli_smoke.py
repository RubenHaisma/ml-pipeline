"""Smoke + rigor tests — the contract CI enforces on every push.

These cover both the *shell* (does the CLI run, is ``--json`` valid, are exit
codes load-bearing) and the *ML rigor that is the point of this repo*:
- the model beats its baseline on ROC-AUC,
- calibration reduces or maintains Brier,
- PSI computes and flags an obviously-shifted feature,
- the full train -> eval -> calibrate -> drift -> infer loop works on CPU.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from ml_pipeline.cli import app

runner = CliRunner()


def _train(out: str) -> dict:
    result = runner.invoke(app, ["train", "configs/churn.yaml", "--out", out, "--json"])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


# ---------------------------------------------------------------- shell


def test_version_json():
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["version"]


def test_doctor_json_is_valid_json():
    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert "ok" in payload and "checks" in payload


def test_bad_config_exits_nonzero():
    result = runner.invoke(app, ["train", "does-not-exist.yaml", "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


def test_eval_without_model_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["eval", "nope", "--out", str(tmp_path), "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


# ---------------------------------------------------------------- ML rigor


def test_model_beats_baseline_on_roc_auc(tmp_path):
    payload = _train(str(tmp_path / "artifacts"))
    m = payload["metrics"]
    assert m["roc_auc"] > m["baseline_roc_auc"]
    assert m["lift_over_baseline"] > 0
    # a dummy prior baseline has no ranking power: ROC-AUC ~= 0.5
    assert abs(m["baseline_roc_auc"] - 0.5) < 0.05
    # the real model should clear a meaningful bar on this signal
    assert m["roc_auc"] > 0.75


def test_eval_matches_train_on_same_holdout(tmp_path):
    out = str(tmp_path / "artifacts")
    trained = _train(out)
    evaled = runner.invoke(app, ["eval", "churn-gb", "--out", out, "--json"])
    assert evaled.exit_code == 0, evaled.stdout
    # eval recomputes on the persisted split, so it must match train exactly
    assert json.loads(evaled.stdout)["metrics"] == trained["metrics"]


def test_calibration_reduces_or_maintains_brier(tmp_path):
    out = str(tmp_path / "artifacts")
    _train(out)
    cal = runner.invoke(app, ["calibrate", "churn-gb", "--out", out, "--json"])
    assert cal.exit_code == 0, cal.stdout
    payload = json.loads(cal.stdout)
    # calibration must not make probabilities worse (small tolerance for noise)
    assert payload["brier_after"] <= payload["brier_before"] + 1e-3
    assert payload["reliability_after"]  # reliability bins are populated


def test_psi_flags_an_obviously_shifted_feature(tmp_path):
    drift = runner.invoke(
        app,
        ["drift", "--reference", "configs/churn.yaml", "--shift", "0.8", "--json"],
    )
    assert drift.exit_code == 0, drift.stdout
    payload = json.loads(drift.stdout)
    # a large shift must flag at least one feature; tenure is shifted hard
    assert payload["n_drifted"] >= 1
    assert "tenure_months" in payload["drifted_features"]
    tenure = next(f for f in payload["features"] if f["feature"] == "tenure_months")
    assert tenure["psi"] >= payload["threshold"]
    # KS gives statistical evidence too for the numeric feature
    assert tenure["ks_pvalue"] < 0.05


def test_no_shift_means_no_drift():
    drift = runner.invoke(
        app,
        ["drift", "--reference", "configs/churn.yaml", "--shift", "0.0", "--json"],
    )
    assert drift.exit_code == 0, drift.stdout
    # with zero shift, reference vs current (only the seed differs) should be stable
    assert json.loads(drift.stdout)["n_drifted"] == 0


def test_full_loop_train_eval_calibrate_drift_infer(tmp_path):
    out = str(tmp_path / "artifacts")
    _train(out)

    assert runner.invoke(app, ["eval", "churn-gb", "--out", out, "--json"]).exit_code == 0
    assert runner.invoke(app, ["calibrate", "churn-gb", "--out", out, "--json"]).exit_code == 0
    assert (
        runner.invoke(
            app, ["drift", "--reference", "configs/churn.yaml", "--shift", "0.5", "--json"]
        ).exit_code
        == 0
    )

    example = json.dumps(
        {
            "tenure_months": 2,
            "monthly_charges": 110.0,
            "total_charges": 220.0,
            "support_calls": 8,
            "logins_last_30d": 1,
            "contract": "month_to_month",
            "payment_method": "card",
            "region": "randstad",
        }
    )
    inferred = runner.invoke(
        app, ["infer", "churn-gb", "--out", out, "--json-input", example, "--json"]
    )
    assert inferred.exit_code == 0, inferred.stdout
    payload = json.loads(inferred.stdout)
    assert payload["ok"] is True
    assert 0.0 <= payload["churn_probability"] <= 1.0
    assert payload["churn_label"] in (0, 1)


def test_infer_default_example_works(tmp_path):
    out = str(tmp_path / "artifacts")
    _train(out)
    inferred = runner.invoke(app, ["infer", "churn-gb", "--out", out, "--json"])
    assert inferred.exit_code == 0, inferred.stdout
    assert json.loads(inferred.stdout)["ok"] is True
