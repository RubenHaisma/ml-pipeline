.PHONY: install up down fmt lint test doctor train eval calibrate drift demo repro readme summary notebooks clean

install:  ## uv sync with dev extras
	uv sync --extra dev

up:  ## bring up MLflow on :5050
	docker compose up -d

down:  ## stop services (volumes preserved)
	docker compose down

fmt:  ## ruff format
	uv run ruff format src tests scripts

lint:  ## ruff check
	uv run ruff check src tests scripts

test:  ## pytest smoke suite
	uv run pytest

doctor:  ## environment readiness check
	uv run mlp doctor

train:  ## train the reference config
	uv run mlp train configs/churn.yaml

eval:  ## recompute holdout metrics vs baseline
	uv run mlp eval churn-gb

calibrate:  ## calibrate probabilities, report Brier before/after
	uv run mlp calibrate churn-gb

drift:  ## PSI + KS drift report against a shifted current set
	uv run mlp drift --reference configs/churn.yaml --shift 0.5

demo: train  ## the full rigor loop: train -> eval -> calibrate -> drift -> infer
	uv run mlp eval churn-gb
	uv run mlp calibrate churn-gb
	uv run mlp drift --reference configs/churn.yaml --shift 0.5
	uv run mlp infer churn-gb

repro:  ## prove training is deterministic (same seed → identical metrics)
	uv run python scripts/check_repro.py -- uv run mlp train configs/churn.yaml

readme:  ## run the README's ci-test commands so the docs can't go stale
	uv run python scripts/test_readme.py

summary:  ## train + print the markdown metrics summary CI posts to the run page
	uv run mlp train configs/churn.yaml --json | uv run python scripts/ci_report.py

notebooks:  ## execute every marimo notebook headless (fails on a dead cell)
	@for nb in notebooks/*.py; do echo "running $$nb"; uv run python "$$nb" || exit 1; done

clean:
	rm -rf artifacts mlruns mlartifacts mlflow.db metrics.json .pytest_cache .ruff_cache
