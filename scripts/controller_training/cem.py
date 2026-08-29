"""Deterministic, resumable standard-library Cross-Entropy Method."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from random import Random
from typing import cast

CEM_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Bounds and initial distribution for one scalar policy parameter."""

    name: str
    minimum: float
    maximum: float
    initial: float
    initial_deviation: float
    minimum_deviation: float = 1e-4

    def __post_init__(self) -> None:
        values = (self.minimum, self.maximum, self.initial, self.initial_deviation, self.minimum_deviation)
        if not self.name.strip():
            raise ValueError("parameter name cannot be empty")
        if not all(isfinite(value) for value in values):
            raise ValueError(f"parameter {self.name} values must be finite")
        if self.minimum >= self.maximum:
            raise ValueError(f"parameter {self.name} minimum must be below maximum")
        if not self.minimum <= self.initial <= self.maximum:
            raise ValueError(f"parameter {self.name} initial value is outside its bounds")
        if self.initial_deviation <= 0.0 or self.minimum_deviation <= 0.0:
            raise ValueError(f"parameter {self.name} deviations must be positive")

    def clamp(self, value: float) -> float:
        """Clamp a sampled value to this parameter's legal interval."""
        return min(max(value, self.minimum), self.maximum)


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """Ordered scalar parameters that make candidate vectors stable."""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("parameter space cannot be empty")
        names = tuple(spec.name for spec in self.specs)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def initial_mean(self) -> tuple[float, ...]:
        return tuple(spec.initial for spec in self.specs)

    @property
    def initial_deviation(self) -> tuple[float, ...]:
        return tuple(spec.initial_deviation for spec in self.specs)

    def mapping(self, values: tuple[float, ...]) -> dict[str, float]:
        """Map a candidate vector onto dataclass keyword arguments."""
        if len(values) != len(self.specs):
            raise ValueError("candidate vector has the wrong length")
        return {spec.name: value for spec, value in zip(self.specs, values, strict=True)}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One sampled parameter vector within a generation."""

    index: int
    values: tuple[float, ...]

    def to_dict(self, space: ParameterSpace) -> dict[str, float]:
        """Return named candidate values."""
        return space.mapping(self.values)


@dataclass(frozen=True, slots=True)
class CEMConfig:
    """Population and update settings for a deterministic CEM run."""

    population_size: int = 48
    elite_count: int = 8
    generations: int = 20
    optimizer_seed: int = 590_112
    smoothing: float = 0.70

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("CEM population must contain at least two candidates")
        if self.elite_count < 1 or self.elite_count >= self.population_size:
            raise ValueError("CEM elite count must be between one and population size")
        if self.generations < 1:
            raise ValueError("CEM generations must be positive")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("CEM smoothing must be in (0, 1]")


class CEMOptimizer:
    """Sample candidates, update from elites, and checkpoint exact RNG state."""

    def __init__(self, *, space: ParameterSpace, config: CEMConfig) -> None:
        self.space = space
        self.config = config
        self.generation = 0
        self.mean = space.initial_mean
        self.deviation = space.initial_deviation
        self._random = Random(config.optimizer_seed)

    @property
    def complete(self) -> bool:
        return self.generation >= self.config.generations

    def sample_population(self) -> tuple[Candidate, ...]:
        """Sample the current generation without mutating its distribution."""
        if self.complete:
            raise RuntimeError("CEM optimization is already complete")
        return tuple(
            Candidate(
                index=index,
                values=tuple(
                    spec.clamp(self._random.gauss(mean, deviation))
                    for spec, mean, deviation in zip(self.space.specs, self.mean, self.deviation, strict=True)
                ),
            )
            for index in range(self.config.population_size)
        )

    def update(self, elites: tuple[Candidate, ...]) -> None:
        """Update distribution moments from the fully evaluated elites."""
        if len(elites) != self.config.elite_count:
            raise ValueError(f"CEM update requires exactly {self.config.elite_count} elites")
        if any(len(candidate.values) != len(self.space.specs) for candidate in elites):
            raise ValueError("elite candidate vector has the wrong length")

        next_mean: list[float] = []
        next_deviation: list[float] = []
        for parameter_index, spec in enumerate(self.space.specs):
            values = tuple(candidate.values[parameter_index] for candidate in elites)
            elite_mean = sum(values) / len(values)
            elite_deviation = sqrt(sum((value - elite_mean) ** 2 for value in values) / len(values))
            smoothed_mean = self.mean[parameter_index] + self.config.smoothing * (
                elite_mean - self.mean[parameter_index]
            )
            smoothed_deviation = self.deviation[parameter_index] + self.config.smoothing * (
                elite_deviation - self.deviation[parameter_index]
            )
            next_mean.append(spec.clamp(smoothed_mean))
            next_deviation.append(max(spec.minimum_deviation, smoothed_deviation))
        self.mean = tuple(next_mean)
        self.deviation = tuple(next_deviation)
        self.generation += 1

    def update_from_ranking(
        self,
        *,
        ranked_population: tuple[Candidate, ...],
        ranked_elites: tuple[Candidate, ...],
        generation_best_score: tuple[float, ...],
    ) -> None:
        """Adapt the shared search interface while preserving positional update()."""
        del ranked_population, generation_best_score
        self.update(ranked_elites)

    def save_checkpoint(
        self,
        path: Path,
        *,
        best_candidate: Candidate,
        metrics: dict[str, object],
    ) -> Path:
        """Atomically persist the next distribution and exact RNG state."""
        record = {
            "schema_version": CEM_CHECKPOINT_SCHEMA_VERSION,
            "generation": self.generation,
            "optimizer_seed": self.config.optimizer_seed,
            "parameter_names": list(self.space.names),
            "distribution_mean": list(self.mean),
            "distribution_deviation": list(self.deviation),
            "best_parameter_vector": best_candidate.to_dict(self.space),
            "best_candidate_index": best_candidate.index,
            "metrics": metrics,
            "random_state": _lists_from_tuples(self._random.getstate()),
        }
        encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(encoded, encoding="utf-8")
        temporary_path.replace(path)
        return path

    @classmethod
    def from_checkpoint(cls, *, space: ParameterSpace, config: CEMConfig, path: Path) -> CEMOptimizer:
        """Resume from the last complete generation without resampling it."""
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(raw, dict):
            raise ValueError("invalid CEM checkpoint schema")
        record = cast(dict[str, object], raw)
        if record.get("schema_version") != CEM_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("invalid CEM checkpoint schema")
        if record.get("optimizer_kind") not in (None, "cem"):
            raise ValueError("CEM checkpoint belongs to a different optimizer")
        if record.get("optimizer_seed") != config.optimizer_seed:
            raise ValueError("CEM checkpoint optimizer seed differs")
        if record.get("parameter_names") != list(space.names):
            raise ValueError("CEM checkpoint parameter space differs")
        optimizer = cls(space=space, config=config)
        optimizer.generation = _integer(record.get("generation"), name="generation")
        optimizer.mean = _float_vector(record.get("distribution_mean"), length=len(space.specs), name="mean")
        optimizer.deviation = _float_vector(
            record.get("distribution_deviation"), length=len(space.specs), name="deviation"
        )
        random_state = _tuples_from_lists(record.get("random_state"))
        optimizer._random.setstate(cast(tuple[int, tuple[int, ...], float | None], random_state))
        return optimizer


def _lists_from_tuples(value: object) -> object:
    if isinstance(value, tuple):
        return [_lists_from_tuples(item) for item in cast(tuple[object, ...], value)]
    return value


def _tuples_from_lists(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuples_from_lists(item) for item in cast(list[object], value))
    return value


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"CEM checkpoint {name} must be an integer")
    return value


def _float_vector(value: object, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"CEM checkpoint {name} has the wrong length")
    items = cast(list[object], value)
    if len(items) != length:
        raise ValueError(f"CEM checkpoint {name} has the wrong length")
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and isfinite(float(item)) for item in items
    ):
        raise ValueError(f"CEM checkpoint {name} must contain finite numbers")
    return tuple(float(cast(int | float, item)) for item in items)
