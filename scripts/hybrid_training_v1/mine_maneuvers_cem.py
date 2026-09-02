#!/usr/bin/env python3
"""Mine short aggressive Formula 110 maneuvers with a smoke-safe CEM loop."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from f110_offline import ScriptedManeuverController, TrialConfig, controller_factory_from_module, run_trials

DEFAULT_OUTPUT = Path("artifacts/maneuvers/best_maneuvers.json")


@dataclass(frozen=True, slots=True)
class Candidate:
    segments: tuple[tuple[int, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {"segments": [list(segment) for segment in self.segments]}


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    score: float
    candidate: Candidate
    result: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "candidate": self.candidate.to_dict(), "result": self.result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=110, help="trial seed or RNG seed")
    parser.add_argument("--start-progress", type=float, default=None, help="optional privileged progress start")
    parser.add_argument("--seconds", type=float, default=5.0, help="short-window simulated seconds")
    parser.add_argument("--segments", type=int, default=5, help="action segments per candidate")
    parser.add_argument("--population", type=int, default=36, help="candidate count per generation")
    parser.add_argument("--generations", type=int, default=18, help="CEM generations")
    parser.add_argument("--elite-fraction", type=float, default=0.25, help="fraction used to update the distribution")
    parser.add_argument("--baseline-module", default="controllers.hybrid_track_policy")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="run a tiny validation loop")
    parser.add_argument("--dry-run", action="store_true", help="alias for --smoke")
    parser.add_argument("--confirm-full", action="store_true", help="required unless --smoke/--dry-run is used")
    return parser.parse_args()


def mine(args: argparse.Namespace) -> list[ScoredCandidate]:
    smoke = bool(args.smoke or args.dry_run)
    if not smoke and not args.confirm_full:
        raise SystemExit("error: full CEM requires --confirm-full; use --smoke for a tiny validation run")
    population = 3 if smoke else int(args.population)
    generations = 1 if smoke else int(args.generations)
    rng = random.Random(int(args.seed))
    baseline_factory = controller_factory_from_module(str(args.baseline_module))
    baseline = run_trials(
        baseline_factory,
        (TrialConfig(seed=int(args.seed), seconds=float(args.seconds), start_progress_m=args.start_progress),),
    )[0]
    duration_mean = [18.0] * int(args.segments)
    duration_std = [9.0] * int(args.segments)
    throttle_mean = [0.78] * int(args.segments)
    throttle_std = [0.35] * int(args.segments)
    steer_mean = [0.20] * int(args.segments)
    steer_std = [0.55] * int(args.segments)
    best: list[ScoredCandidate] = []
    for generation in range(generations):
        scored: list[ScoredCandidate] = []
        for _ in range(population):
            candidate = _sample_candidate(
                rng,
                duration_mean=duration_mean,
                duration_std=duration_std,
                throttle_mean=throttle_mean,
                throttle_std=throttle_std,
                steer_mean=steer_mean,
                steer_std=steer_std,
            )
            result = run_trials(
                lambda candidate=candidate: ScriptedManeuverController(candidate.segments, baseline_factory()),
                (TrialConfig(seed=int(args.seed), seconds=float(args.seconds), start_progress_m=args.start_progress),),
            )[0]
            scored.append(ScoredCandidate(_fitness(result.to_dict(), baseline.to_dict()), candidate, result.to_dict()))
        scored.sort(key=lambda item: item.score, reverse=True)
        best = sorted([*best, *scored[:5]], key=lambda item: item.score, reverse=True)[:10]
        elites = scored[: max(1, int(len(scored) * float(args.elite_fraction)))]
        duration_mean, duration_std = _update_int_distribution([elite.candidate.segments for elite in elites], 0)
        throttle_mean, throttle_std = _update_float_distribution([elite.candidate.segments for elite in elites], 1)
        steer_mean, steer_std = _update_float_distribution([elite.candidate.segments for elite in elites], 2)
        print(f"generation {generation + 1}/{generations}: best={scored[0].score:.3f}")
    return best


def _sample_candidate(
    rng: random.Random,
    *,
    duration_mean: list[float],
    duration_std: list[float],
    throttle_mean: list[float],
    throttle_std: list[float],
    steer_mean: list[float],
    steer_std: list[float],
) -> Candidate:
    segments = []
    for index in range(len(duration_mean)):
        duration = int(max(3, min(55, round(rng.gauss(duration_mean[index], duration_std[index])))))
        throttle = max(-1.0, min(1.0, rng.gauss(throttle_mean[index], throttle_std[index])))
        steer = max(-1.0, min(1.0, rng.gauss(steer_mean[index], steer_std[index])))
        segments.append((duration, throttle, steer))
    return Candidate(tuple(segments))


def _fitness(result: dict[str, object], baseline: dict[str, object]) -> float:
    if result.get("ok") is not True:
        return -10_000.0
    damage = float(result["final_damage"])
    progress_delta = float(result["raw_distance_m"])
    baseline_delta = float(baseline.get("raw_distance_m", 0.0))
    speed = float(result["max_speed_mps"])
    survived = bool(result["survived"])
    wall_seconds = float(result["wall_contact_seconds"])
    score = progress_delta * 8.0 + max(0.0, progress_delta - baseline_delta) * 14.0 + speed * 2.5
    score += min(wall_seconds, 0.45) * 15.0
    if survived:
        score += 60.0
    if damage >= 1.0:
        score -= 5000.0
    elif damage > 0.85:
        score -= 800.0 * (damage - 0.85)
    else:
        score += max(0.0, 0.40 - damage) * 25.0
    if progress_delta < baseline_delta * 0.70:
        score -= 150.0
    return score


def _update_int_distribution(
    elite_segments: list[tuple[tuple[int, float, float], ...]],
    value_index: int,
) -> tuple[list[float], list[float]]:
    means, stds = _update_float_distribution(elite_segments, value_index)
    return means, [max(2.0, std) for std in stds]


def _update_float_distribution(
    elite_segments: list[tuple[tuple[int, float, float], ...]],
    value_index: int,
) -> tuple[list[float], list[float]]:
    segment_count = len(elite_segments[0])
    means: list[float] = []
    stds: list[float] = []
    for index in range(segment_count):
        values = [float(candidate[index][value_index]) for candidate in elite_segments]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        stds.append(max(0.04, variance**0.5))
    return means, stds


def main() -> None:
    args = parse_args()
    best = mine(args)
    payload = {
        "maneuvers": [item.to_dict() for item in best],
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {len(best)} maneuver templates to {args.output}")


if __name__ == "__main__":
    main()
