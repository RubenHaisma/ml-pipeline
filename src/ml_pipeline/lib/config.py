"""Typed config loading. One YAML file in, one validated dataclass out.

Kept deliberately small: a dataclass for the train config shape, a loader that
fails loud with a :class:`CliError` instead of a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ml_pipeline.output import CliError


@dataclass(slots=True)
class TrainConfig:
    """Everything a churn training run needs, resolved from YAML + defaults.

    ``dataset`` is ``synthetic`` (the deterministic generator used in CI) or a
    path to a CSV. When a CSV is supplied, ``target`` names the label column.
    """

    name: str
    dataset: str = "synthetic"
    target: str = "churn"
    model: str = "gradient_boosting"
    params: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 4000
    test_size: float = 0.25
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        p = Path(path)
        if not p.is_file():
            raise CliError(f"config not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - exercised via CLI
            raise CliError(f"invalid yaml in {p}: {exc}") from exc
        if "name" not in raw:
            raise CliError(f"config {p} is missing required key: name")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})
