#!/usr/bin/env python3
"""Tune hybrid controller parameters with smoke-safe stochastic search."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from f110_offline import TrialConfig, run_trials

DEFAULT_OUTPUT = Path("artifacts/policies/best_policy.json")
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "heading_gain": (0.010, 0.034),
    "center_gain": (0.060, 0.260),
    "lookahead_gain": (0.010, 0.075),
    "yaw_damping_gain": (0.0005, 0.0050),
    "speed_kp": (0.080, 0.240),
    "speed_brake_kp": (0.080, 0.320),
    "hard_brake_speed_error": (1.40, 4.20),
    "large_heading_error_degrees": (28.0, 58.0),
    "front_wall_emergency_m": (0.85, 2.25),
    "side_wall_bias_gain": (0.0, 0.095),
    "stuck_speed_mps": (0.30, 1.00),
    "stuck_seconds": (0.55, 1.80),
    "target_speed_scale": (0.85, 1.65),
    "target_speed_bias_mps": (-1.0, 5.5),
    "curve_speed_penalty_scale": (-0.35, 1.2),
    "straight_speed_bonus_mps": (0.0, 5.5),
    "straight_curvature_threshold": (0.04, 0.75),
    "min_target_speed_mps": (3.5, 8.0),
    "max_target_speed_mps": (14.0, 28.0),
    "front_wall_speed_scale": (1.9, 5.2),
    "large_heading_target_speed_mps": (2.5, 8.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110, 2026])
    parser.add_argument("--robustness-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--trials", type=int, default=96, help="candidate policies to evaluate")
    parser.add_argument("--seed", type=int, default=110, help="optimizer RNG seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="run only a tiny validation search")
    parser.add_argument("--confirm-full", action="store_true", help="required unless --smoke is used")
    parser.add_argument("--compact-output", action="store_true", help="write best params/results without full history")
    return parser.parse_args()


def tune(args: argparse.Namespace) -> dict[str, object]:
    if not args.smoke and not args.confirm_full:
        raise SystemExit("error: full tuning requires --confirm-full; use --smoke for a tiny validation run")
    from controllers.hybrid_track_policy import HybridTrackController
    from controllers.hybrid_track_policy_data import DEFAULT_GAINS

    rng = random.Random(int(args.seed))
    trials = 2 if args.smoke else int(args.trials)
    seconds = min(float(args.seconds), 3.0) if args.smoke else float(args.seconds)
    seeds = tuple(int(seed) for seed in [*args.seeds, *args.robustness_seeds])
    best_score = -1.0e18
    best_params: dict[str, float] = {}
    best_record: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    for trial_index in range(trials):
        params = _sample_params(rng, DEFAULT_GAINS, exploration=1.0 if trial_index else 0.0)
        results = run_trials(
            lambda params=params: HybridTrackController(params),
            tuple(TrialConfig(seed=seed, seconds=seconds) for seed in seeds),
        )
        score = _policy_fitness([result.to_dict() for result in results], primary_seed_count=len(args.seeds))
        record = {"score": score, "params": params, "results": [result.to_dict() for result in results]}
        history.append(record)
        if score > best_score:
            best_score = score
            best_params = params
            best_record = record
        print(f"trial {trial_index + 1}/{trials}: score={score:.3f}, best={best_score:.3f}")
    if args.compact_output:
        return {"best_score": best_score, "best_params": best_params, "best_record": best_record}
    return {"best_score": best_score, "best_params": best_params, "history": history}


def _sample_params(rng: random.Random, defaults: dict[str, float], *, exploration: float) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, (low, high) in PARAM_BOUNDS.items():
        default = defaults[name]
        if exploration <= 0.0:
            params[name] = default
            continue
        span = high - low
        value = default + rng.gauss(0.0, span * 0.22 * exploration)
        if rng.random() < 0.20:
            value = rng.uniform(low, high)
        params[name] = max(low, min(high, value))
    return params


def _policy_fitness(results: list[dict[str, object]], *, primary_seed_count: int) -> float:
    if not results or any(result.get("ok") is not True for result in results):
        return -1.0e12
    primary = results[:primary_seed_count]
    mean_distance = sum(float(result["raw_distance_m"]) for result in primary) / len(primary)
    score = mean_distance * 10.0
    for result in results:
        damage = float(result["final_damage"])
        if damage >= 1.0 or not bool(result["survived"]):
            score -= 1_000_000.0
        elif damage > 0.85:
            score -= 4_000.0 * (damage - 0.85)
        score += int(result["lap_count"]) * 30.0
        score += float(result["max_speed_mps"]) * 1.25
    if len(results) > primary_seed_count:
        robust = results[primary_seed_count:]
        score += 2.0 * sum(float(result["raw_distance_m"]) for result in robust) / len(robust)
    return score


def main() -> None:
    args = parse_args()
    payload = tune(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote best policy parameters to {args.output}")


if __name__ == "__main__":
    main()
