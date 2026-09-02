#!/usr/bin/env python3
"""Mine aggressive short-horizon sector trajectories with CEM."""

from __future__ import annotations

import argparse
import atexit
import json
import multiprocessing
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from f110_offline import (
    ControllerFactory,
    HeadlessRaceSession,
    ScriptedManeuverController,
    TrialConfig,
    controller_factory_from_module,
)

CONTROL_HZ = 60.0
DEFAULT_OUTPUT = Path("artifacts/sectors/sector_trajectories.json")
_WORKER_SESSION: HeadlessRaceSession | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_BASELINE_FACTORY: ControllerFactory | None = None
_WORKER_SECTOR_COUNT = 0


@dataclass(frozen=True, slots=True)
class Candidate:
    """A fixed-horizon normalized action sequence."""

    segments: tuple[tuple[int, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {"segments": [list(segment) for segment in self.segments]}


@dataclass(frozen=True, slots=True)
class SectorSpec:
    """One privileged sector start state to mine from."""

    start_progress_m: float
    start_speed_mps: float
    start_lateral_offset_m: float = 0.0
    start_heading_error_degrees: float = 0.0


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate with its simulator result and scalar score."""

    score: float
    candidate: Candidate
    result: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "candidate": self.candidate.to_dict(), "result": self.result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-progresses", nargs="+", type=float, default=[0.0])
    parser.add_argument("--start-speeds", nargs="+", type=float, default=[0.0, 18.0, 30.0])
    parser.add_argument("--lateral-offsets", nargs="+", type=float, default=[0.0])
    parser.add_argument("--heading-errors", nargs="+", type=float, default=[0.0])
    parser.add_argument("--duration", type=float, default=1.5, help="seconds per mined sector")
    parser.add_argument("--segments", type=int, default=8, help="fixed action segments per candidate")
    parser.add_argument("--population", type=int, default=48, help="candidate count per generation")
    parser.add_argument("--generations", type=int, default=24, help="CEM generations per sector")
    parser.add_argument("--elite-fraction", type=float, default=0.22)
    parser.add_argument("--keep", type=int, default=12, help="top candidates to keep per sector")
    parser.add_argument("--trace-keep", type=int, default=3, help="top candidates to rerun with per-tick traces")
    parser.add_argument("--seed", type=int, default=110, help="optimizer RNG seed")
    parser.add_argument("--trial-seed", type=int, default=110, help="simulator seed")
    parser.add_argument("--baseline-module", default="controllers.hybrid_track_policy")
    parser.add_argument("--damage-limit", type=float, default=0.98)
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes for independent sectors")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="run a tiny validation job")
    parser.add_argument("--confirm-full", action="store_true", help="required unless --smoke is used")
    return parser.parse_args()


def mine(args: argparse.Namespace) -> dict[str, object]:
    if not args.smoke and not args.confirm_full:
        raise SystemExit("error: full sector mining requires --confirm-full; use --smoke for a tiny validation run")
    rng = random.Random(int(args.seed))
    sectors = _sector_specs(args)
    population = 4 if args.smoke else int(args.population)
    generations = 1 if args.smoke else int(args.generations)
    workers = max(1, int(args.workers))
    run_args = argparse.Namespace(**vars(args))
    run_args.population = population
    run_args.generations = generations
    if workers > 1 and len(sectors) > 1:
        return _mine_parallel(run_args, sectors, worker_count=min(workers, len(sectors)))
    baseline_factory = controller_factory_from_module(str(run_args.baseline_module))
    payload_sectors: list[dict[str, object]] = []
    with HeadlessRaceSession() as session:
        for sector_index, sector in enumerate(sectors, start=1):
            payload_sectors.append(
                _mine_sector_with_session(
                    session=session,
                    args=run_args,
                    sector=sector,
                    sector_index=sector_index,
                    sector_count=len(sectors),
                    rng=random.Random(rng.randrange(2**63)),
                    baseline_factory=baseline_factory,
                )
            )
            _write_checkpoint(run_args, payload_sectors, complete=False)

    return _payload(run_args, payload_sectors, complete=True)


def _mine_parallel(
    args: argparse.Namespace,
    sectors: tuple[SectorSpec, ...],
    *,
    worker_count: int,
) -> dict[str, object]:
    rng = random.Random(int(args.seed))
    tasks = tuple((sector_index, sector, rng.randrange(2**63)) for sector_index, sector in enumerate(sectors, start=1))
    context = multiprocessing.get_context("spawn")
    print(f"mining {len(sectors)} sectors with {worker_count} worker processes", flush=True)
    with context.Pool(
        processes=worker_count,
        initializer=_initialize_worker,
        initargs=(args, len(sectors)),
    ) as pool:
        payload_by_index: dict[int, dict[str, object]] = {}
        for sector_index, payload_sector in pool.imap_unordered(_mine_sector_worker, tasks):
            payload_by_index[sector_index] = payload_sector
            _write_checkpoint(args, _ordered_payload_sectors(payload_by_index), complete=False)
            print(f"checkpointed {len(payload_by_index)}/{len(sectors)} sectors", flush=True)
    return _payload(args, _ordered_payload_sectors(payload_by_index), complete=True)


def _payload(
    args: argparse.Namespace,
    payload_sectors: list[dict[str, object]],
    *,
    complete: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "complete": complete,
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "sectors": payload_sectors,
    }


def _ordered_payload_sectors(payload_by_index: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    return [payload_by_index[index] for index in sorted(payload_by_index)]


def _write_checkpoint(args: argparse.Namespace, payload_sectors: list[dict[str, object]], *, complete: bool) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_payload(args, payload_sectors, complete=complete), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _initialize_worker(args: argparse.Namespace, sector_count: int) -> None:
    global _WORKER_ARGS, _WORKER_BASELINE_FACTORY, _WORKER_SECTOR_COUNT, _WORKER_SESSION
    _WORKER_ARGS = args
    _WORKER_SECTOR_COUNT = sector_count
    _WORKER_BASELINE_FACTORY = controller_factory_from_module(str(args.baseline_module))
    _WORKER_SESSION = HeadlessRaceSession()
    atexit.register(_close_worker_session)


def _close_worker_session() -> None:
    global _WORKER_SESSION
    if _WORKER_SESSION is None:
        return
    _WORKER_SESSION.close()
    _WORKER_SESSION = None


def _mine_sector_worker(task: tuple[int, SectorSpec, int]) -> tuple[int, dict[str, object]]:
    sector_index, sector, rng_seed = task
    if _WORKER_SESSION is None or _WORKER_ARGS is None or _WORKER_BASELINE_FACTORY is None:
        raise RuntimeError("sector worker was not initialized")
    return (
        sector_index,
        _mine_sector_with_session(
            session=_WORKER_SESSION,
            args=_WORKER_ARGS,
            sector=sector,
            sector_index=sector_index,
            sector_count=_WORKER_SECTOR_COUNT,
            rng=random.Random(rng_seed),
            baseline_factory=_WORKER_BASELINE_FACTORY,
        ),
    )


def _mine_sector_with_session(
    *,
    session: HeadlessRaceSession,
    args: argparse.Namespace,
    sector: SectorSpec,
    sector_index: int,
    sector_count: int,
    rng: random.Random,
    baseline_factory: ControllerFactory,
) -> dict[str, object]:
    baseline = session.run_trial(baseline_factory, _trial_config(args, sector))
    best: list[ScoredCandidate] = []
    throttle_mean = [0.86] * int(args.segments)
    throttle_std = [0.34] * int(args.segments)
    steer_mean = [0.0] * int(args.segments)
    steer_std = [0.56] * int(args.segments)
    durations = _segment_durations(duration_seconds=float(args.duration), segment_count=int(args.segments))
    for generation in range(int(args.generations)):
        scored: list[ScoredCandidate] = []
        for _ in range(int(args.population)):
            candidate = _sample_candidate(
                rng,
                durations=durations,
                throttle_mean=throttle_mean,
                throttle_std=throttle_std,
                steer_mean=steer_mean,
                steer_std=steer_std,
            )
            result = session.run_trial(
                lambda candidate=candidate: ScriptedManeuverController(
                    candidate.segments,
                    baseline_factory(),
                ),
                _trial_config(args, sector),
            )
            score = _fitness(result.to_dict(), baseline.to_dict(), damage_limit=float(args.damage_limit))
            scored.append(ScoredCandidate(score=score, candidate=candidate, result=result.to_dict()))
        scored.sort(key=lambda item: item.score, reverse=True)
        best = sorted([*best, *scored[: int(args.keep)]], key=lambda item: item.score, reverse=True)[: int(args.keep)]
        elites = scored[: max(1, int(len(scored) * float(args.elite_fraction)))]
        throttle_mean, throttle_std = _update_distribution([elite.candidate.segments for elite in elites], 1)
        steer_mean, steer_std = _update_distribution([elite.candidate.segments for elite in elites], 2)
        top = scored[0].result
        print(
            "sector "
            f"{sector_index}/{sector_count} gen {generation + 1}/{int(args.generations)}: "
            f"score={scored[0].score:.2f}, "
            f"dist={float(top['raw_distance_m']):.2f} m, "
            f"final={float(top['final_speed_mps']):.2f} m/s, "
            f"damage={float(top['final_damage']):.3f}",
            flush=True,
        )
    traced_best = _trace_candidates(
        session=session,
        args=args,
        sector=sector,
        candidates=tuple(item.candidate for item in best[: max(0, int(args.trace_keep))]),
        baseline_factory=baseline_factory,
        damage_limit=float(args.damage_limit),
    )
    return {
        "sector": asdict(sector),
        "baseline": baseline.to_dict(),
        "best": [*traced_best, *(candidate.to_dict() for candidate in best[len(traced_best) :])],
    }


def _sector_specs(args: argparse.Namespace) -> tuple[SectorSpec, ...]:
    specs = tuple(
        SectorSpec(
            start_progress_m=float(progress),
            start_speed_mps=float(speed),
            start_lateral_offset_m=float(lateral),
            start_heading_error_degrees=float(heading),
        )
        for progress in args.start_progresses
        for speed in args.start_speeds
        for lateral in args.lateral_offsets
        for heading in args.heading_errors
    )
    if not specs:
        raise ValueError("at least one sector spec is required")
    return specs


def _trial_config(args: argparse.Namespace, sector: SectorSpec, *, record_trace: bool = False) -> TrialConfig:
    return TrialConfig(
        seed=int(args.trial_seed),
        seconds=float(args.duration),
        start_progress_m=sector.start_progress_m,
        start_speed_mps=sector.start_speed_mps,
        start_lateral_offset_m=sector.start_lateral_offset_m,
        start_heading_error_degrees=sector.start_heading_error_degrees,
        record_trace=record_trace,
    )


def _trace_candidates(
    *,
    session: HeadlessRaceSession,
    args: argparse.Namespace,
    sector: SectorSpec,
    candidates: tuple[Candidate, ...],
    baseline_factory: ControllerFactory,
    damage_limit: float,
) -> list[dict[str, object]]:
    traced: list[dict[str, object]] = []
    baseline = session.run_trial(baseline_factory, _trial_config(args, sector))
    for candidate in candidates:
        result = session.run_trial(
            lambda candidate=candidate: ScriptedManeuverController(
                candidate.segments,
                baseline_factory(),
            ),
            _trial_config(args, sector, record_trace=True),
        )
        traced.append(
            ScoredCandidate(
                score=_fitness(result.to_dict(), baseline.to_dict(), damage_limit=damage_limit),
                candidate=candidate,
                result=result.to_dict(),
            ).to_dict()
        )
    return traced


def _segment_durations(*, duration_seconds: float, segment_count: int) -> tuple[int, ...]:
    if duration_seconds <= 0.0:
        raise ValueError("--duration must be positive")
    if segment_count < 1:
        raise ValueError("--segments must be at least one")
    total_ticks = max(segment_count, round(duration_seconds * CONTROL_HZ))
    base_ticks, extra_ticks = divmod(total_ticks, segment_count)
    return tuple(base_ticks + (1 if index < extra_ticks else 0) for index in range(segment_count))


def _sample_candidate(
    rng: random.Random,
    *,
    durations: tuple[int, ...],
    throttle_mean: list[float],
    throttle_std: list[float],
    steer_mean: list[float],
    steer_std: list[float],
) -> Candidate:
    segments: list[tuple[int, float, float]] = []
    for index, duration in enumerate(durations):
        throttle = _sample_control(rng, mean=throttle_mean[index], std=throttle_std[index], low=-1.0, high=1.0)
        steer = _sample_control(rng, mean=steer_mean[index], std=steer_std[index], low=-1.0, high=1.0)
        segments.append((duration, throttle, steer))
    return Candidate(tuple(segments))


def _sample_control(rng: random.Random, *, mean: float, std: float, low: float, high: float) -> float:
    if rng.random() < 0.12:
        return rng.uniform(low, high)
    return max(low, min(high, rng.gauss(mean, std)))


def _fitness(result: dict[str, object], baseline: dict[str, object], *, damage_limit: float) -> float:
    if result.get("ok") is not True:
        return -1.0e9
    progress_m = float(result["raw_distance_m"])
    baseline_m = float(baseline.get("raw_distance_m", 0.0))
    damage = float(result["final_damage"])
    wall_seconds = float(result["wall_contact_seconds"])
    final_speed = float(result.get("final_speed_mps", 0.0))
    max_speed = float(result["max_speed_mps"])
    score = progress_m * 12.0 + max(0.0, progress_m - baseline_m) * 16.0
    score += final_speed * 5.0 + max_speed * 1.5
    score -= damage * 55.0 + wall_seconds * 1.5
    if bool(result["survived"]):
        score += 70.0
    if damage >= damage_limit:
        score -= 25_000.0 * (damage - damage_limit + 0.02)
    if damage >= 1.0 or not bool(result["survived"]):
        score -= 1_000_000.0
    return score


def _update_distribution(
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
        stds.append(max(0.035, variance**0.5))
    return means, stds


def main() -> None:
    args = parse_args()
    payload = mine(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {len(payload['sectors'])} mined sector(s) to {args.output}")


if __name__ == "__main__":
    main()
