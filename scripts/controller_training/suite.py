#!/usr/bin/env python3
"""Evaluate one controller module across a deterministic seed suite."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import cast

from scripts.controller_training.evaluator import (
    SOLO_TRIAL_DEFAULT_SECONDS,
    SoloEvaluator,
    SoloTrialResult,
    controller_factory_from_module,
)
from scripts.controller_training.search import percentile
from scripts.controller_training.seeds import generate_seed_manifest

SUITE_RESULT_SCHEMA_VERSION = 1
DEFAULT_WORKER_COUNT = max(1, (os.cpu_count() or 1) - 1)
_worker_evaluator: SoloEvaluator | None = None


@dataclass(frozen=True, slots=True)
class SuiteTask:
    module: str
    seed: int
    duration_seconds: float


def evaluate_suite(
    *,
    module: str,
    seeds: tuple[int, ...],
    duration_seconds: float = SOLO_TRIAL_DEFAULT_SECONDS,
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> tuple[SoloTrialResult, ...]:
    """Evaluate every seed in isolated controller state on persistent workers."""
    if not seeds:
        raise ValueError("evaluation suite cannot be empty")
    if worker_count < 1:
        raise ValueError("worker count must be positive")
    tasks = tuple(SuiteTask(module=module, seed=seed, duration_seconds=duration_seconds) for seed in seeds)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(_evaluate_suite_task, tasks))


def suite_record(
    *,
    module: str,
    duration_seconds: float,
    results: tuple[SoloTrialResult, ...],
) -> dict[str, object]:
    """Return detailed trials plus stable aggregate promotion metrics."""
    if not results:
        raise ValueError("suite record requires at least one result")
    distances = tuple(result.raw_distance_m for result in results)
    return {
        "schema_version": SUITE_RESULT_SCHEMA_VERSION,
        "record_type": "solo_trial_suite",
        "module": module,
        "duration_seconds": duration_seconds,
        "seeds": [result.seed for result in results],
        "results": [result.to_dict() for result in results],
        "summary": {
            "trial_count": len(results),
            "survival_count": sum(1 for result in results if result.survived),
            "lap_count": sum(1 for result in results if result.lap_count >= 1),
            "clean_count": sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0),
            "damage_trial_count": sum(1 for result in results if result.damage > 0.0),
            "wall_contact_trial_count": sum(1 for result in results if result.wall_contact_seconds > 0.0),
            "worst_distance_m": min(distances),
            "mean_distance_m": fmean(distances),
            "median_distance_m": median(distances),
            "tenth_percentile_distance_m": percentile(distances, 0.10),
            "maximum_speed_mps": max(result.max_speed_mps for result in results),
            "worst_damage": max(result.damage for result in results),
            "total_wall_contact_seconds": sum(result.wall_contact_seconds for result in results),
        },
    }


def _evaluate_suite_task(task: SuiteTask) -> SoloTrialResult:
    global _worker_evaluator
    if _worker_evaluator is None:
        _worker_evaluator = SoloEvaluator()
    return _worker_evaluator.run_trial(
        controller_factory=controller_factory_from_module(task.module),
        seed=task.seed,
        duration_seconds=task.duration_seconds,
    )


def _named_seeds(name: str) -> tuple[int, ...]:
    manifest = generate_seed_manifest()
    suites = {
        "official": manifest.official,
        "training": manifest.training,
        "validation": manifest.validation,
        "soak": manifest.final_soak,
    }
    return suites[name]


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--suite", choices=("official", "training", "validation", "soak"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seconds", type=float, default=SOLO_TRIAL_DEFAULT_SECONDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(arguments)
    if (args.suite is None) == (args.seeds is None):
        parser.error("choose exactly one of --suite or --seeds")
    return args


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    module = str(args.module)
    duration_seconds = float(args.seconds)
    seeds = _named_seeds(str(args.suite)) if args.suite is not None else tuple(cast(list[int], args.seeds))
    results = evaluate_suite(
        module=module,
        seeds=seeds,
        duration_seconds=duration_seconds,
        worker_count=int(args.workers),
    )
    record = suite_record(module=module, duration_seconds=duration_seconds, results=results)
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = cast(Path, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
