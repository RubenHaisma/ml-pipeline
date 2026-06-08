.PHONY: install up down fmt lint test doctor train eval calibrate drift demo clean

install:  ## uv sync with dev extras
	uv sync --extra dev

up:  ## bring up MLflow on :5050
	docker compose up -d

down:  ## stop services (volumes preserved)
	docker compose down

fmt:  ## ruff format
	uv run ruff format src tests

lint:  ## ruff check
	uv run ruff check src tests

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

clean:
	rm -rf artifacts mlruns mlartifacts mlflow.db .pytest_cache .ruff_cache
