"""Deterministic, resumable bounded genetic optimizer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from random import Random
from typing import cast

from scripts.controller_training.cem import Candidate, ParameterSpace

GA_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GAConfig:
    """Population, crossover, mutation, and stagnation settings."""

    population_size: int = 64
    elite_count: int = 12
    generations: int = 40
    optimizer_seed: int = 590_115
    tournament_size: int = 3
    crossover_alpha: float = 0.5
    mutation_probability: float = 0.25
    mutation_scale: float = 0.10
    seeded_fraction: float = 0.25
    stagnation_patience: int = 4
    stagnation_inflation: float = 1.8
    maximum_mutation_scale: float = 0.35

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("GA population must contain at least two candidates")
        if self.elite_count < 1 or self.elite_count >= self.population_size:
            raise ValueError("GA elite count must be between one and population size")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("GA tournament size must be between two and population size")
        if self.generations < 1 or self.stagnation_patience < 1:
            raise ValueError("GA generation and stagnation counts must be positive")
        if self.crossover_alpha < 0.0:
            raise ValueError("GA crossover alpha cannot be negative")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("GA mutation probability must be in [0, 1]")
        if self.mutation_scale <= 0.0 or self.maximum_mutation_scale < self.mutation_scale:
            raise ValueError("GA mutation scales are inconsistent")
        if not 0.0 < self.seeded_fraction <= 1.0:
            raise ValueError("GA seeded fraction must be in (0, 1]")
        if self.stagnation_inflation <= 1.0:
            raise ValueError("GA stagnation inflation must exceed one")


class GeneticOptimizer:
    """Generate bounded offspring and checkpoint exact evolutionary state."""

    def __init__(
        self,
        *,
        space: ParameterSpace,
        config: GAConfig,
        checkpoint_context: dict[str, float] | None = None,
    ) -> None:
        self.space = space
        self.config = config
        self.generation = 0
        self.population: tuple[tuple[float, ...], ...] = ()
        self.elite_values: tuple[tuple[float, ...], ...] = ()
        self.current_mutation_scale = config.mutation_scale
        self.stagnation_counter = 0
        self.best_score: tuple[float, ...] = ()
        self.checkpoint_context = dict(sorted((checkpoint_context or {}).items()))
        self._random = Random(config.optimizer_seed)

    @property
    def complete(self) -> bool:
        return self.generation >= self.config.generations

    def sample_population(self) -> tuple[Candidate, ...]:
        """Return the seeded first generation or offspring of the ranked parents."""
        if self.complete:
            raise RuntimeError("GA optimization is already complete")
        if self.generation == 0:
            values = self._seeded_population()
        else:
            if len(self.population) != self.config.population_size:
                raise RuntimeError("GA ranked population is unavailable")
            if len(self.elite_values) != self.config.elite_count:
                raise RuntimeError("GA elite population is unavailable")
            children = tuple(self._offspring() for _ in range(self.config.population_size - self.config.elite_count))
            values = (*self.elite_values, *children)
        return tuple(Candidate(index=index, values=candidate) for index, candidate in enumerate(values))

    def _seeded_population(self) -> tuple[tuple[float, ...], ...]:
        seeded_count = max(
            1, min(self.config.population_size, round(self.config.population_size * self.config.seeded_fraction))
        )
        population = [self.space.initial_mean]
        while len(population) < seeded_count:
            population.append(
                tuple(
                    spec.clamp(
                        self._random.gauss(spec.initial, self.config.mutation_scale * (spec.maximum - spec.minimum))
                    )
                    for spec in self.space.specs
                )
            )
        while len(population) < self.config.population_size:
            population.append(tuple(self._random.uniform(spec.minimum, spec.maximum) for spec in self.space.specs))
        return tuple(population)

    def _offspring(self) -> tuple[float, ...]:
        first = self._tournament_parent()
        second = self._tournament_parent()
        child: list[float] = []
        for spec, first_value, second_value in zip(self.space.specs, first, second, strict=True):
            distance = abs(first_value - second_value)
            lower = min(first_value, second_value) - self.config.crossover_alpha * distance
            upper = max(first_value, second_value) + self.config.crossover_alpha * distance
            value = self._random.uniform(lower, upper)
            if self._random.random() < self.config.mutation_probability:
                value += self._random.gauss(0.0, self.current_mutation_scale * (spec.maximum - spec.minimum))
            child.append(spec.clamp(value))
        return tuple(child)

    def _tournament_parent(self) -> tuple[float, ...]:
        # Population is stored best-first, so the smallest sampled rank wins.
        ranks = self._random.sample(range(len(self.population)), self.config.tournament_size)
        return self.population[min(ranks)]

    def update_from_ranking(
        self,
        *,
        ranked_population: tuple[Candidate, ...],
        ranked_elites: tuple[Candidate, ...],
        generation_best_score: tuple[float, ...],
    ) -> None:
        """Retain ranked parents/elites and update deterministic stagnation state."""
        if len(ranked_population) != self.config.population_size:
            raise ValueError(f"GA update requires exactly {self.config.population_size} ranked candidates")
        if len(ranked_elites) != self.config.elite_count:
            raise ValueError(f"GA update requires exactly {self.config.elite_count} elites")
        if any(len(candidate.values) != len(self.space.specs) for candidate in (*ranked_population, *ranked_elites)):
            raise ValueError("GA candidate vector has the wrong length")
        if not generation_best_score or not all(isfinite(value) for value in generation_best_score):
            raise ValueError("GA best score must contain finite values")

        self.population = tuple(candidate.values for candidate in ranked_population)
        self.elite_values = tuple(candidate.values for candidate in ranked_elites)
        if not self.best_score or generation_best_score > self.best_score:
            self.best_score = generation_best_score
            self.stagnation_counter = 0
            self.current_mutation_scale = self.config.mutation_scale
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.config.stagnation_patience:
                self.current_mutation_scale = min(
                    self.config.maximum_mutation_scale,
                    self.current_mutation_scale * self.config.stagnation_inflation,
                )
                self.stagnation_counter = 0
        self.generation += 1

    def save_checkpoint(
        self,
        path: Path,
        *,
        best_candidate: Candidate,
        metrics: dict[str, object],
    ) -> Path:
        """Atomically persist the evaluated population and exact RNG state."""
        record = {
            "schema_version": GA_CHECKPOINT_SCHEMA_VERSION,
            "optimizer_kind": "ga",
            "generation": self.generation,
            "optimizer_seed": self.config.optimizer_seed,
            "optimizer_config": asdict(self.config),
            "parameter_names": list(self.space.names),
            "checkpoint_context": self.checkpoint_context,
            "population": [list(values) for values in self.population],
            "elite_values": [list(values) for values in self.elite_values],
            "mutation_scale": self.current_mutation_scale,
            "stagnation_counter": self.stagnation_counter,
            "best_score": list(self.best_score),
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
    def from_checkpoint(
        cls,
        *,
        space: ParameterSpace,
        config: GAConfig,
        path: Path,
        checkpoint_context: dict[str, float] | None = None,
    ) -> GeneticOptimizer:
        """Resume from the last complete generation without regenerating it."""
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(raw, dict):
            raise ValueError("invalid GA checkpoint schema")
        record = cast(dict[str, object], raw)
        if record.get("schema_version") != GA_CHECKPOINT_SCHEMA_VERSION or record.get("optimizer_kind") != "ga":
            raise ValueError("invalid GA checkpoint schema")
        if record.get("optimizer_seed") != config.optimizer_seed:
            raise ValueError("GA checkpoint optimizer seed differs")
        if record.get("parameter_names") != list(space.names):
            raise ValueError("GA checkpoint parameter space differs")
        expected_context = dict(sorted((checkpoint_context or {}).items()))
        if record.get("checkpoint_context") != expected_context:
            raise ValueError("GA checkpoint fixed parameters differ")
        _validate_config(record.get("optimizer_config"), config)

        optimizer = cls(space=space, config=config, checkpoint_context=expected_context)
        optimizer.generation = _integer(record.get("generation"), name="generation")
        optimizer.population = _float_matrix(
            record.get("population"),
            rows=config.population_size,
            columns=len(space.specs),
            name="population",
        )
        optimizer.elite_values = _float_matrix(
            record.get("elite_values"),
            rows=config.elite_count,
            columns=len(space.specs),
            name="elite_values",
        )
        optimizer.current_mutation_scale = _finite_float(record.get("mutation_scale"), name="mutation_scale")
        optimizer.stagnation_counter = _integer(record.get("stagnation_counter"), name="stagnation_counter")
        optimizer.best_score = _float_vector(record.get("best_score"), name="best_score")
        if not 0 <= optimizer.generation <= config.generations:
            raise ValueError("GA checkpoint generation is outside the configured run")
        if not config.mutation_scale <= optimizer.current_mutation_scale <= config.maximum_mutation_scale:
            raise ValueError("GA checkpoint mutation scale is outside configured bounds")
        if optimizer.stagnation_counter < 0:
            raise ValueError("GA checkpoint stagnation counter cannot be negative")
        for vector in (*optimizer.population, *optimizer.elite_values):
            if any(not spec.minimum <= value <= spec.maximum for spec, value in zip(space.specs, vector, strict=True)):
                raise ValueError("GA checkpoint candidate is outside the parameter space")
        random_state = _tuples_from_lists(record.get("random_state"))
        optimizer._random.setstate(cast(tuple[int, tuple[int, ...], float | None], random_state))
        return optimizer


def _validate_config(value: object, config: GAConfig) -> None:
    if not isinstance(value, dict):
        raise ValueError("GA checkpoint optimizer config is invalid")
    stored = cast(dict[str, object], value)
    expected = asdict(config)
    # The generation limit may be extended when resuming a completed run.
    for name, expected_value in expected.items():
        if name != "generations" and stored.get(name) != expected_value:
            raise ValueError(f"GA checkpoint optimizer config differs for {name}")
    stored_generations = _integer(stored.get("generations"), name="optimizer_config.generations")
    if config.generations < stored_generations:
        raise ValueError("GA generation limit cannot shrink on resume")


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
        raise ValueError(f"GA checkpoint {name} must be an integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
        raise ValueError(f"GA checkpoint {name} must be finite")
    return float(value)


def _float_vector(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"GA checkpoint {name} must contain values")
    return tuple(_finite_float(item, name=name) for item in cast(list[object], value))


def _float_matrix(
    value: object,
    *,
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"GA checkpoint {name} has the wrong row count")
    matrix: list[tuple[float, ...]] = []
    for row in cast(list[object], value):
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"GA checkpoint {name} has the wrong column count")
        matrix.append(tuple(_finite_float(item, name=name) for item in cast(list[object], row)))
    return tuple(matrix)
