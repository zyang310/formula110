"""Append-only strict JSON experiment records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

EXPERIMENT_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One retained controller experiment and its promotion decision."""

    experiment_id: str
    date: str
    controller: str
    artifact: str
    seeds: tuple[int, ...]
    metrics: dict[str, object]
    decision: str
    next_step: str

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment id cannot be empty")
        if not self.controller.strip():
            raise ValueError("experiment controller cannot be empty")

    def to_dict(self) -> dict[str, object]:
        """Return a versioned JSON-compatible record."""
        return {
            "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
            "record_type": "controller_experiment",
            "experiment_id": self.experiment_id,
            "date": self.date,
            "controller": self.controller,
            "artifact": self.artifact,
            "seeds": list(self.seeds),
            "metrics": self.metrics,
            "decision": self.decision,
            "next_step": self.next_step,
        }


@dataclass(frozen=True, slots=True)
class ExperimentJournal:
    """Append-only JSONL journal stored with ignored training artifacts."""

    path: Path

    def append(self, record: ExperimentRecord) -> None:
        """Append one strict JSON record and reject duplicate experiment ids."""
        existing_ids = {existing.experiment_id for existing in self.read_all()}
        if record.experiment_id in existing_ids:
            raise ValueError(f"experiment id already exists: {record.experiment_id}")
        encoded = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def read_all(self) -> tuple[ExperimentRecord, ...]:
        """Read and validate every retained record."""
        if not self.path.exists():
            return ()
        records: list[ExperimentRecord] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                raw = cast(object, json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid experiment JSON on line {line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"invalid experiment record on line {line_number}")
            record = cast(dict[str, object], raw)
            if record.get("schema_version") != EXPERIMENT_RECORD_SCHEMA_VERSION:
                raise ValueError(f"invalid experiment record on line {line_number}")
            seeds = record.get("seeds")
            metrics = record.get("metrics")
            if not isinstance(seeds, list):
                raise ValueError(f"invalid experiment seeds on line {line_number}")
            seed_values = cast(list[object], seeds)
            if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seed_values):
                raise ValueError(f"invalid experiment seeds on line {line_number}")
            if not isinstance(metrics, dict):
                raise ValueError(f"invalid experiment metrics on line {line_number}")
            metric_values = cast(dict[str, object], metrics)
            records.append(
                ExperimentRecord(
                    experiment_id=_string(record.get("experiment_id"), name="experiment_id"),
                    date=_string(record.get("date"), name="date"),
                    controller=_string(record.get("controller"), name="controller"),
                    artifact=_string(record.get("artifact"), name="artifact"),
                    seeds=tuple(cast(int, seed) for seed in seed_values),
                    metrics=metric_values,
                    decision=_string(record.get("decision"), name="decision"),
                    next_step=_string(record.get("next_step"), name="next_step"),
                )
            )
        return tuple(records)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"experiment record {name} must be a string")
    return value
