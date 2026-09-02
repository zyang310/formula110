#!/usr/bin/env python3
"""Run local Formula 110 single-car controller trials in batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from f110_offline import TrialConfig, controller_factory_from_module, run_trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="controller module or file, such as controllers.race_faster")
    parser.add_argument("--function", default="control", help="controller function name (default: control)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[110, 2026], help="trial seeds")
    parser.add_argument("--seconds", type=float, default=30.0, help="simulated seconds per trial")
    parser.add_argument("--race-index", type=int, default=1, help="seeded race index")
    parser.add_argument("--start-progress", type=float, default=None, help="override start progress distance in meters")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module_reference = Path(args.module) if str(args.module).endswith(".py") else str(args.module)
    configs = tuple(
        TrialConfig(
            seed=seed,
            seconds=float(args.seconds),
            race_index=int(args.race_index),
            start_progress_m=args.start_progress,
        )
        for seed in args.seeds
    )
    results = run_trials(
        controller_factory_from_module(module_reference, function_name=str(args.function)),
        configs,
    )
    payload = {
        "module": args.module,
        "function": args.function,
        "seconds": float(args.seconds),
        "results": [result.to_dict() for result in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
        return
    for result in results:
        if not result.ok:
            print(f"seed {result.seed}: ERROR {result.error}")
            continue
        first_lap = f"{result.first_lap_time_seconds:.3f}s" if result.first_lap_time_seconds is not None else "none"
        best_lap = f"{result.best_lap_time_seconds:.3f}s" if result.best_lap_time_seconds is not None else "none"
        print(
            f"seed {result.seed}: {result.raw_distance_m:.2f} m, "
            f"{result.partial_laps:.4f} laps, lap_count={result.lap_count}, "
            f"survived={result.survived}, damage={result.final_damage:.4f}, "
            f"wall={result.wall_contact_seconds:.3f}s, top={result.max_speed_mps:.2f} m/s, "
            f"first_lap={first_lap}, best_lap={best_lap}"
        )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
