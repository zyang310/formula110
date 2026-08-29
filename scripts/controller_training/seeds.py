"""Immutable seed suites for controller training and promotion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import cast

SEED_MANIFEST_SCHEMA_VERSION = 1
OFFICIAL_SEEDS = (110, 2026)
SEED_MINIMUM = 1
SEED_MAXIMUM = 99_999
TRAINING_SEED_COUNT = 28
VALIDATION_SEED_COUNT = 12
FINAL_SOAK_SEED_COUNT = 100
TRAINING_RANDOM_SEED = 590_110
FINAL_SOAK_RANDOM_SEED = 590_111


@dataclass(frozen=True, slots=True)
class SeedManifest:
    """Non-overlapping deterministic suites used by the controller project."""

    official: tuple[int, ...]
    training: tuple[int, ...]
    validation: tuple[int, ...]
    final_soak: tuple[int, ...]

    def __post_init__(self) -> None:
        suites = (self.official, self.training, self.validation, self.final_soak)
        combined = tuple(seed for suite in suites for seed in suite)
        if any(seed < SEED_MINIMUM or seed > SEED_MAXIMUM for seed in combined):
            raise ValueError("seed manifest values must be between 1 and 99,999")
        if len(set(combined)) != len(combined):
            raise ValueError("seed manifest suites must not overlap")
        if len(self.official) != len(OFFICIAL_SEEDS):
            raise ValueError("seed manifest requires two official seeds")
        if len(self.training) != TRAINING_SEED_COUNT:
            raise ValueError("seed manifest requires 28 training seeds")
        if len(self.validation) != VALIDATION_SEED_COUNT:
            raise ValueError("seed manifest requires 12 validation seeds")
        if len(self.final_soak) != FINAL_SOAK_SEED_COUNT:
            raise ValueError("seed manifest requires 100 final-soak seeds")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible manifest."""
        return {
            "schema_version": SEED_MANIFEST_SCHEMA_VERSION,
            "official": list(self.official),
            "training": list(self.training),
            "validation": list(self.validation),
            "final_soak": list(self.final_soak),
        }


def generate_seed_manifest() -> SeedManifest:
    """Generate the immutable suites from their documented RNG seeds."""
    excluded: set[int] = set(OFFICIAL_SEEDS)
    generated = _unique_random_seeds(
        random_seed=TRAINING_RANDOM_SEED,
        count=TRAINING_SEED_COUNT + VALIDATION_SEED_COUNT,
        excluded=excluded,
    )
    training = generated[:TRAINING_SEED_COUNT]
    validation = generated[TRAINING_SEED_COUNT:]
    excluded.update(generated)
    final_soak = _unique_random_seeds(
        random_seed=FINAL_SOAK_RANDOM_SEED,
        count=FINAL_SOAK_SEED_COUNT,
        excluded=excluded,
    )
    return SeedManifest(
        official=OFFICIAL_SEEDS,
        training=training,
        validation=validation,
        final_soak=final_soak,
    )


def write_seed_manifest(path: Path, manifest: SeedManifest | None = None) -> Path:
    """Write one stable manifest, refusing to replace different seed suites."""
    resolved_manifest = generate_seed_manifest() if manifest is None else manifest
    record = json.dumps(resolved_manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        existing = load_seed_manifest(path)
        if existing != resolved_manifest:
            raise ValueError(f"existing seed manifest differs: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record, encoding="utf-8")
    return path


def load_seed_manifest(path: Path) -> SeedManifest:
    """Load and validate a versioned seed manifest."""
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("seed manifest must be a JSON object")
    record = cast(dict[str, object], raw)
    if record.get("schema_version") != SEED_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported seed manifest schema version")
    return SeedManifest(
        official=_integer_tuple(record.get("official"), name="official"),
        training=_integer_tuple(record.get("training"), name="training"),
        validation=_integer_tuple(record.get("validation"), name="validation"),
        final_soak=_integer_tuple(record.get("final_soak"), name="final_soak"),
    )


def _unique_random_seeds(*, random_seed: int, count: int, excluded: set[int]) -> tuple[int, ...]:
    random = Random(random_seed)
    generated: list[int] = []
    seen = set(excluded)
    while len(generated) < count:
        candidate = random.randint(SEED_MINIMUM, SEED_MAXIMUM)
        if candidate in seen:
            continue
        seen.add(candidate)
        generated.append(candidate)
    return tuple(generated)


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"seed manifest {name} must be an integer list")
    items = cast(list[object], value)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise ValueError(f"seed manifest {name} must be an integer list")
    return tuple(cast(int, item) for item in items)
