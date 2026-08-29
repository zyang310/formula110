#!/usr/bin/env python3
"""Tune preview-controller presets with deterministic, resumable optimizers."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Literal, Protocol, cast

from scripts.controller_training.cem import Candidate, CEMConfig, CEMOptimizer, ParameterSpace, ParameterSpec
from scripts.controller_training.evaluator import SOLO_TRIAL_DEFAULT_SECONDS, SoloEvaluator, SoloTrialResult
from scripts.controller_training.genetic import GAConfig, GeneticOptimizer
from scripts.controller_training.seeds import generate_seed_manifest, write_seed_manifest

from controllers.minimum_viable import MINIMUM_VIABLE_PARAMETERS
from controllers.preview_controller import ControllerParameters, PreviewController
from controllers.race_faster import RACE_FASTER_PARAMETERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "controller-search"
DEFAULT_WORKER_COUNT = max(1, (os.cpu_count() or 1) - 1)
ROTATING_SEED_COUNT = 6
SearchPreset = Literal["minimum", "faster", "faster-line", "faster-line-v2-probe", "faster-line-v2", "faster-line-v3"]
OptimizerKind = Literal["cem", "ga"]
ObjectiveKind = Literal["improved", "improved-v2"]
Score = tuple[float, ...]

# Per-trial safety budget for `improved_score`. Damage is capped well below the 1.0
# elimination threshold; the distance penalties keep a gradient inside the budget.
INCIDENT_BUDGET_DAMAGE = 0.25
INCIDENT_BUDGET_CONTACT_S = 1.50
DAMAGE_DISTANCE_PENALTY_M = 120.0
CONTACT_DISTANCE_PENALTY_M = 6.0

FASTER_LINE_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    phase_aware_racing_line=True,
)

FASTER_LINE_V2_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    curvature_offset_compensation=1.0,
    curvature_heading_compensation=0.0,
    preview_line_compensation=1.0,
    wall_balance_line_compensation=1.0,
    line_turn_sensitivity=0.045,
    line_target_slew_per_tick=0.02,
    line_clearance_m=1.8,
)


# The promoted 595.98 m vector, with the two signals it could not use.  Five of
# its genes finished on a bound, so v3 exists to move the bounds rather than to
# search longer: the line clamp, the two line floors, and the curvature ratio
# whose ceiling was forced by the distance-inflated speed signal.
FASTER_LINE_V3_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    maximum_racing_line_offset_ratio=0.80,
    # Redefined by the new speed signal: a local curvature in 1/m that peaks near
    # 0.17, not the old distance-scaled ratio that peaked near 1.0.  Calibrated
    # on a seed-110 trace of the promoted vector as the value whose `curvature`
    # distribution best matches what that vector actually ran.
    curvature_lateral_ratio=0.155,
)


class Optimizer(Protocol):
    """Common lifecycle used by deterministic controller optimizers."""

    generation: int

    @property
    def complete(self) -> bool: ...

    def sample_population(self) -> tuple[Candidate, ...]: ...

    def update_from_ranking(
        self,
        *,
        ranked_population: tuple[Candidate, ...],
        ranked_elites: tuple[Candidate, ...],
        generation_best_score: Score,
    ) -> None: ...

    def save_checkpoint(
        self,
        path: Path,
        *,
        best_candidate: Candidate,
        metrics: dict[str, object],
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """Picklable candidate work assigned to one persistent Panda process."""

    candidate: Candidate
    parameter_names: tuple[str, ...]
    base_parameters: ControllerParameters
    seeds: tuple[int, ...]
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One candidate's results and lexicographic score."""

    candidate: Candidate
    results: tuple[SoloTrialResult, ...]
    score: Score


_worker_evaluator: SoloEvaluator | None = None


def minimum_parameter_space(base: ControllerParameters = MINIMUM_VIABLE_PARAMETERS) -> ParameterSpace:
    """Return conservative bounds around the promoted minimum preset."""
    return ParameterSpace(
        (
            ParameterSpec("heading_steer_gain", 0.80, 1.06, base.heading_steer_gain, 0.055, 0.003),
            ParameterSpec("center_steer_gain", 0.42, 0.64, base.center_steer_gain, 0.045, 0.003),
            ParameterSpec("yaw_damping_gain", 0.13, 0.28, base.yaw_damping_gain, 0.032, 0.002),
            ParameterSpec("curvature_heading_degrees", 45.0, 62.0, base.curvature_heading_degrees, 3.4, 0.20),
            ParameterSpec("curvature_lateral_ratio", 0.48, 0.70, base.curvature_lateral_ratio, 0.045, 0.003),
            ParameterSpec("straight_target_speed_mps", 10.4, 12.4, base.straight_target_speed_mps, 0.40, 0.03),
            ParameterSpec("corner_target_speed_mps", 5.9, 7.5, base.corner_target_speed_mps, 0.32, 0.02),
            ParameterSpec("steering_speed_reduction", 0.20, 0.36, base.steering_speed_reduction, 0.035, 0.002),
            ParameterSpec("yaw_speed_reduction", 0.14, 0.30, base.yaw_speed_reduction, 0.035, 0.002),
            ParameterSpec("front_brake_start_m", 7.0, 9.5, base.front_brake_start_m, 0.45, 0.03),
            ParameterSpec("brake_gain", 0.22, 0.42, base.brake_gain, 0.040, 0.003),
            ParameterSpec("maximum_forward_throttle", 0.72, 1.0, base.maximum_forward_throttle, 0.055, 0.004),
        )
    )


def faster_parameter_space(base: ControllerParameters = RACE_FASTER_PARAMETERS) -> ParameterSpace:
    """Return the closed `faster` search space, retained to document that run.

    The search ended at generation 84 and is not resumable: it optimized a braking
    policy that coasting replaced. Initials are clamped into the bounds because
    `race_faster.py` now holds a `faster-line` vector that falls outside several of
    them, and an unclamped initial would make this function raise on import.
    """
    return ParameterSpace(
        (
            ParameterSpec(
                "heading_steer_gain", 0.84, 1.16, _bounded(base.heading_steer_gain, 0.84, 1.16), 0.065, 0.003
            ),
            ParameterSpec("center_steer_gain", 0.40, 0.70, _bounded(base.center_steer_gain, 0.40, 0.70), 0.060, 0.003),
            ParameterSpec("yaw_damping_gain", 0.10, 0.28, _bounded(base.yaw_damping_gain, 0.10, 0.28), 0.040, 0.002),
            ParameterSpec(
                "racing_line_offset_ratio", 0.0, 0.24, _bounded(base.racing_line_offset_ratio, 0.0, 0.24), 0.050, 0.003
            ),
            ParameterSpec(
                "curvature_heading_degrees",
                44.0,
                76.0,
                _bounded(base.curvature_heading_degrees, 44.0, 76.0),
                6.0,
                0.25,
            ),
            ParameterSpec(
                "curvature_lateral_ratio", 0.54, 0.95, _bounded(base.curvature_lateral_ratio, 0.54, 0.95), 0.080, 0.004
            ),
            ParameterSpec(
                "straight_target_speed_mps",
                13.0,
                17.5,
                _bounded(base.straight_target_speed_mps, 13.0, 17.5),
                0.80,
                0.04,
            ),
            ParameterSpec(
                "corner_target_speed_mps", 7.5, 11.5, _bounded(base.corner_target_speed_mps, 7.5, 11.5), 0.70, 0.04
            ),
            ParameterSpec(
                "steering_speed_reduction",
                0.06,
                0.26,
                _bounded(base.steering_speed_reduction, 0.06, 0.26),
                0.045,
                0.003,
            ),
            ParameterSpec(
                "yaw_speed_reduction", 0.05, 0.24, _bounded(base.yaw_speed_reduction, 0.05, 0.24), 0.040, 0.003
            ),
            ParameterSpec("front_brake_start_m", 4.5, 7.5, _bounded(base.front_brake_start_m, 4.5, 7.5), 0.60, 0.04),
            ParameterSpec("side_slow_start_m", 0.75, 1.50, _bounded(base.side_slow_start_m, 0.75, 1.50), 0.14, 0.01),
            ParameterSpec("side_speed_floor", 0.50, 0.90, _bounded(base.side_speed_floor, 0.50, 0.90), 0.08, 0.005),
            ParameterSpec("brake_gain", 0.17, 0.42, _bounded(base.brake_gain, 0.17, 0.42), 0.050, 0.003),
        )
    )


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def faster_line_parameter_space(base: ControllerParameters = FASTER_LINE_BASE_PARAMETERS) -> ParameterSpace:
    """Tune the coasting speed schedule and a phase-aware outside-inside-outside line.

    Initials are explicit rather than read from ``base``. The baked vector comes from
    the closed ``faster`` search, which ran against a braking policy and against a
    17.5 m/s ceiling, so its speed and damping values are no longer good starting
    points. ``side_slow_start_m`` and ``side_speed_floor`` are dropped because AVOID
    never fires on the training seeds and varying them changed nothing measurable.
    """
    del base
    return ParameterSpace(
        (
            ParameterSpec("straight_target_speed_mps", 18.0, 26.0, 21.0, 1.6, 0.08),
            ParameterSpec("corner_target_speed_mps", 11.0, 17.0, 13.5, 1.2, 0.06),
            ParameterSpec("throttle_gain", 0.18, 0.90, 0.45, 0.15, 0.008),
            ParameterSpec("front_brake_start_m", 4.0, 14.0, 8.0, 1.8, 0.06),
            ParameterSpec("steering_speed_reduction", 0.0, 0.35, 0.10, 0.07, 0.004),
            ParameterSpec("yaw_speed_reduction", 0.0, 0.30, 0.08, 0.06, 0.004),
            ParameterSpec("curvature_heading_degrees", 45.0, 90.0, 64.0, 8.0, 0.30),
            ParameterSpec("curvature_lateral_ratio", 0.60, 1.40, 0.95, 0.14, 0.007),
            ParameterSpec("heading_steer_gain", 0.75, 1.30, 0.94, 0.10, 0.005),
            ParameterSpec("center_steer_gain", 0.10, 0.55, 0.30, 0.09, 0.004),
            ParameterSpec("wall_balance_gain", 0.0, 0.45, 0.20, 0.10, 0.005),
            ParameterSpec("steer_slew_per_tick", 0.06, 0.20, 0.10, 0.025, 0.0015),
            ParameterSpec("racing_line_offset_ratio", 0.0, 0.45, 0.10, 0.10, 0.006),
            ParameterSpec("racing_line_entry_offset_ratio", 0.0, 0.45, 0.10, 0.10, 0.006),
            ParameterSpec("racing_line_exit_offset_ratio", 0.0, 0.45, 0.10, 0.10, 0.006),
        )
    )


def faster_line_v2_probe_parameter_space(
    base: ControllerParameters = FASTER_LINE_V2_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 19-gene falsification space for the pose-invariant line."""
    return ParameterSpace(
        (
            ParameterSpec("straight_target_speed_mps", 18.0, 26.0, base.straight_target_speed_mps, 1.6, 0.08),
            ParameterSpec("corner_target_speed_mps", 13.0, 24.0, base.corner_target_speed_mps, 2.0, 0.10),
            ParameterSpec("throttle_gain", 0.30, 2.00, base.throttle_gain, 0.30, 0.015),
            ParameterSpec("front_brake_start_m", 4.0, 14.0, base.front_brake_start_m, 1.8, 0.06),
            ParameterSpec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            ParameterSpec("curvature_lateral_ratio", 0.20, 1.00, 0.45, 0.16, 0.01),
            ParameterSpec("heading_steer_gain", 0.60, 1.30, base.heading_steer_gain, 0.12, 0.006),
            ParameterSpec("center_steer_gain", 0.10, 0.90, base.center_steer_gain, 0.16, 0.008),
            ParameterSpec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            ParameterSpec("racing_line_offset_ratio", 0.0, 0.65, base.racing_line_offset_ratio, 0.12, 0.008),
            ParameterSpec(
                "racing_line_entry_offset_ratio", 0.0, 0.65, base.racing_line_entry_offset_ratio, 0.12, 0.008
            ),
            ParameterSpec("racing_line_exit_offset_ratio", 0.0, 0.65, base.racing_line_exit_offset_ratio, 0.12, 0.008),
            ParameterSpec("curvature_offset_compensation", 0.0, 1.0, base.curvature_offset_compensation, 0.20, 0.01),
            ParameterSpec("curvature_heading_compensation", 0.0, 1.0, base.curvature_heading_compensation, 0.20, 0.01),
            ParameterSpec("preview_line_compensation", 0.0, 1.0, base.preview_line_compensation, 0.20, 0.01),
            ParameterSpec("wall_balance_line_compensation", 0.0, 1.0, base.wall_balance_line_compensation, 0.20, 0.01),
            # The phase signal is a local curvature in 1/m, which peaks at about
            # 0.16 on this track; the old [0.03, 0.40] box spent over half its
            # width on values that switch the racing line off entirely.
            ParameterSpec("line_turn_sensitivity", 0.010, 0.150, base.line_turn_sensitivity, 0.030, 0.0015),
            ParameterSpec("line_target_slew_per_tick", 0.005, 0.10, base.line_target_slew_per_tick, 0.02, 0.001),
            ParameterSpec("line_clearance_m", 0.0, 2.5, base.line_clearance_m, 0.50, 0.025),
        )
    )


def faster_line_v3_parameter_space(
    base: ControllerParameters = FASTER_LINE_V3_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 16-gene space that unpins the v2 winner's bounds.

    Two genes leave the space entirely: `track_shape_preview` cancels the car's
    pose algebraically, so `curvature_offset_compensation` and
    `curvature_heading_compensation` no longer affect anything once
    `pose_invariant_speed_curvature` is on.
    """
    return ParameterSpace(
        (
            ParameterSpec("straight_target_speed_mps", 18.0, 28.0, base.straight_target_speed_mps, 1.6, 0.08),
            ParameterSpec("corner_target_speed_mps", 13.0, 26.0, base.corner_target_speed_mps, 2.0, 0.10),
            ParameterSpec("throttle_gain", 0.30, 2.00, base.throttle_gain, 0.30, 0.015),
            ParameterSpec("front_brake_start_m", 4.0, 14.0, base.front_brake_start_m, 1.8, 0.06),
            ParameterSpec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            # The speed scalar is now a local curvature in 1/m peaking near 0.16.
            ParameterSpec("curvature_lateral_ratio", 0.02, 0.30, base.curvature_lateral_ratio, 0.06, 0.003),
            ParameterSpec("heading_steer_gain", 0.40, 1.30, base.heading_steer_gain, 0.12, 0.006),
            # Floor dropped to zero: the v2 winner sat on the old 0.10 floor, and
            # after the line-frame debias the preview term carries the line.
            ParameterSpec("center_steer_gain", 0.0, 0.90, base.center_steer_gain, 0.16, 0.008),
            ParameterSpec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            # The v2 winner pinned entry against its 0.65 clamp, so the clamp is
            # searched here and the three ratios are bounded by its ceiling.
            ParameterSpec(
                "maximum_racing_line_offset_ratio", 0.65, 0.90, base.maximum_racing_line_offset_ratio, 0.08, 0.004
            ),
            ParameterSpec("racing_line_offset_ratio", 0.0, 0.90, base.racing_line_offset_ratio, 0.14, 0.008),
            ParameterSpec(
                "racing_line_entry_offset_ratio", 0.0, 0.90, base.racing_line_entry_offset_ratio, 0.14, 0.008
            ),
            ParameterSpec("racing_line_exit_offset_ratio", 0.0, 0.90, base.racing_line_exit_offset_ratio, 0.14, 0.008),
            # Floor dropped: the v2 winner sat on the old 0.010 floor.
            ParameterSpec("line_turn_sensitivity", 0.002, 0.150, base.line_turn_sensitivity, 0.030, 0.0015),
            ParameterSpec("line_target_slew_per_tick", 0.005, 0.15, base.line_target_slew_per_tick, 0.02, 0.001),
            ParameterSpec("line_clearance_m", 0.0, 3.0, base.line_clearance_m, 0.50, 0.025),
        )
    )


def faster_line_v2_parameter_space(
    base: ControllerParameters = FASTER_LINE_V2_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the final 17-gene GA space with line-frame springs fixed."""
    probe = faster_line_v2_probe_parameter_space(base)
    fixed = {"preview_line_compensation", "wall_balance_line_compensation"}
    return ParameterSpace(
        tuple(
            ParameterSpec(
                spec.name,
                spec.minimum,
                spec.maximum,
                _bounded(float(getattr(base, spec.name)), spec.minimum, spec.maximum),
                spec.initial_deviation,
                spec.minimum_deviation,
            )
            for spec in probe.specs
            if spec.name not in fixed
        )
    )


def rotating_training_seeds(training: tuple[int, ...], generation: int) -> tuple[int, ...]:
    """Select six deterministic training seeds, wrapping around the suite."""
    if len(training) < ROTATING_SEED_COUNT:
        raise ValueError("training suite must contain at least six seeds")
    offset = generation * ROTATING_SEED_COUNT % len(training)
    return tuple(training[(offset + index) % len(training)] for index in range(ROTATING_SEED_COUNT))


def minimum_score(results: tuple[SoloTrialResult, ...]) -> Score:
    """Rank safe laps first, then clean trials, worst distance, and mean distance."""
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    safe_lap_count = sum(
        1
        for result in results
        if result.survived and result.lap_count >= 1 and result.damage == 0.0 and result.wall_contact_seconds == 0.0
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    distances = tuple(result.raw_distance_m for result in results)
    return (float(safe_lap_count), float(clean_count), min(distances), fmean(distances))


def improved_score(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank survival and lap completion, then robust improvement inside a safety budget.

    The budget is counted per trial rather than summed, so a small bounded incident
    no longer outranks every distance term. `race_faster` is gated on surviving and
    completing a lap, not on running perfectly clean; that stricter rule belongs to
    `minimum_score`.
    """
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    improvements = tuple(result.raw_distance_m / baseline_distances[result.seed] - 1.0 for result in results)
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    penalized_distances = tuple(
        result.raw_distance_m
        - DAMAGE_DISTANCE_PENALTY_M * result.damage
        - CONTACT_DISTANCE_PENALTY_M * result.wall_contact_seconds
        for result in results
    )
    return (
        float(sum(1 for result in results if result.survived)),
        float(sum(1 for result in results if result.lap_count >= 1)),
        float(within_budget),
        min(improvements),
        percentile(improvements, 0.10),
        median(improvements),
        fmean(penalized_distances),
    )


def improved_score_v2(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Keep hard safety tiers and replace the worst-case cliff with penalized p10."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    penalized_distances = tuple(
        result.raw_distance_m
        - DAMAGE_DISTANCE_PENALTY_M * result.damage
        - CONTACT_DISTANCE_PENALTY_M * result.wall_contact_seconds
        for result in results
    )
    mean_penalized = fmean(penalized_distances)
    robust_distance = mean_penalized - (mean_penalized - percentile(penalized_distances, 0.10))
    return (
        float(sum(1 for result in results if result.survived)),
        float(sum(1 for result in results if result.lap_count >= 1)),
        float(within_budget),
        robust_distance,
    )


def percentile(values: tuple[float, ...], probability: float) -> float:
    """Return a linearly interpolated inclusive percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def run_search(
    *,
    preset: SearchPreset,
    artifact_root: Path,
    config: CEMConfig | GAConfig,
    optimizer_kind: OptimizerKind = "cem",
    objective_kind: ObjectiveKind = "improved",
    seed_checkpoint: Path | None = None,
    duration_seconds: float = SOLO_TRIAL_DEFAULT_SECONDS,
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> Candidate:
    """Run or resume every configured generation and return the best candidate."""
    if worker_count < 1:
        raise ValueError("worker count must be positive")
    if optimizer_kind == "cem" and not isinstance(config, CEMConfig):
        raise TypeError("CEM search requires CEMConfig")
    if optimizer_kind == "ga" and not isinstance(config, GAConfig):
        raise TypeError("GA search requires GAConfig")
    manifest = generate_seed_manifest()
    write_seed_manifest(artifact_root.parent / "seed-manifest.json", manifest)
    base_parameters, space = preset_configuration(preset, seed_checkpoint=seed_checkpoint)
    checkpoint_context = _checkpoint_context(preset, base_parameters)
    checkpoint_path = artifact_root / "checkpoint.json"
    if checkpoint_path.exists():
        _validate_search_checkpoint(
            checkpoint_path,
            preset=preset,
            optimizer_kind=optimizer_kind,
            objective_kind=objective_kind,
        )
        optimizer = _load_optimizer(
            optimizer_kind=optimizer_kind,
            space=space,
            config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_context=checkpoint_context,
        )
        best_candidate, best_score = _checkpoint_best(checkpoint_path, space)
        archive_generation_checkpoint(
            checkpoint_path=checkpoint_path,
            artifact_root=artifact_root,
            generation=optimizer.generation,
        )
    else:
        optimizer = _new_optimizer(
            optimizer_kind=optimizer_kind,
            space=space,
            config=config,
            checkpoint_context=checkpoint_context,
        )
        best_candidate = Candidate(index=-1, values=space.initial_mean)
        score_length = 4 if preset == "minimum" or objective_kind == "improved-v2" else 7
        best_score = tuple(float("-inf") for _ in range(score_length))

    if optimizer.complete:
        return best_candidate

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        baseline_distances = _baseline_distances(
            executor=executor,
            seeds=manifest.training,
            duration_seconds=duration_seconds,
        )
        while not optimizer.complete:
            selection_seeds = rotating_training_seeds(manifest.training, optimizer.generation)
            population = optimizer.sample_population()
            batch_evaluations = _evaluate_candidates(
                executor=executor,
                candidates=population,
                parameter_names=space.names,
                base_parameters=base_parameters,
                seeds=selection_seeds,
                duration_seconds=duration_seconds,
                preset=preset,
                objective_kind=objective_kind,
                baseline_distances=baseline_distances,
            )
            ranked_batch = tuple(sorted(batch_evaluations, key=lambda evaluation: evaluation.score, reverse=True))
            selected = tuple(evaluation.candidate for evaluation in ranked_batch[: config.elite_count])
            full_evaluations = _evaluate_candidates(
                executor=executor,
                candidates=selected,
                parameter_names=space.names,
                base_parameters=base_parameters,
                seeds=manifest.training,
                duration_seconds=duration_seconds,
                preset=preset,
                objective_kind=objective_kind,
                baseline_distances=baseline_distances,
            )
            ranked_full = tuple(sorted(full_evaluations, key=lambda evaluation: evaluation.score, reverse=True))
            generation_best = ranked_full[0]
            optimizer.update_from_ranking(
                ranked_population=tuple(evaluation.candidate for evaluation in ranked_batch),
                ranked_elites=tuple(evaluation.candidate for evaluation in ranked_full),
                generation_best_score=generation_best.score,
            )
            if generation_best.score > best_score:
                best_candidate = generation_best.candidate
                best_score = generation_best.score
                best_results = generation_best.results
            else:
                best_results = ()
            metrics = _checkpoint_metrics(
                preset=preset,
                selection_seeds=selection_seeds,
                best_score=best_score,
                generation_best=generation_best,
                promoted_results=best_results,
                duration_seconds=duration_seconds,
                optimizer_kind=optimizer_kind,
                objective_kind=objective_kind,
                diversity_metrics=_diversity_metrics(
                    space=space,
                    ranked_batch=ranked_batch,
                    ranked_elites=ranked_full,
                    elite_count=config.elite_count,
                ),
            )
            optimizer.save_checkpoint(
                checkpoint_path,
                best_candidate=best_candidate,
                metrics=metrics,
            )
            archive_generation_checkpoint(
                checkpoint_path=checkpoint_path,
                artifact_root=artifact_root,
                generation=optimizer.generation,
            )
            print(
                json.dumps(
                    {
                        "generation": optimizer.generation,
                        "best_score": list(best_score),
                        "generation_best_score": list(generation_best.score),
                        "best_parameters": best_candidate.to_dict(space),
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
    return best_candidate


def _new_optimizer(
    *,
    optimizer_kind: OptimizerKind,
    space: ParameterSpace,
    config: CEMConfig | GAConfig,
    checkpoint_context: dict[str, float],
) -> Optimizer:
    if optimizer_kind == "cem" and isinstance(config, CEMConfig):
        return CEMOptimizer(space=space, config=config)
    if optimizer_kind == "ga" and isinstance(config, GAConfig):
        return GeneticOptimizer(space=space, config=config, checkpoint_context=checkpoint_context)
    raise TypeError("optimizer kind and configuration do not match")


def _load_optimizer(
    *,
    optimizer_kind: OptimizerKind,
    space: ParameterSpace,
    config: CEMConfig | GAConfig,
    checkpoint_path: Path,
    checkpoint_context: dict[str, float],
) -> Optimizer:
    if optimizer_kind == "cem" and isinstance(config, CEMConfig):
        return CEMOptimizer.from_checkpoint(space=space, config=config, path=checkpoint_path)
    if optimizer_kind == "ga" and isinstance(config, GAConfig):
        return GeneticOptimizer.from_checkpoint(
            space=space,
            config=config,
            path=checkpoint_path,
            checkpoint_context=checkpoint_context,
        )
    raise TypeError("optimizer kind and configuration do not match")


def _checkpoint_context(preset: SearchPreset, parameters: ControllerParameters) -> dict[str, float]:
    if preset not in ("faster-line-v2", "faster-line-v3"):
        return {}
    return {
        "curvature_heading_degrees": parameters.curvature_heading_degrees,
        "preview_line_compensation": parameters.preview_line_compensation,
        "steer_slew_per_tick": parameters.steer_slew_per_tick,
        "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
        "yaw_speed_reduction": parameters.yaw_speed_reduction,
    }


def _validate_search_checkpoint(
    path: Path,
    *,
    preset: SearchPreset,
    optimizer_kind: OptimizerKind,
    objective_kind: ObjectiveKind,
) -> None:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("invalid search checkpoint")
    record = cast(dict[str, object], raw)
    recorded_optimizer = record.get("optimizer_kind", "cem")
    if recorded_optimizer != optimizer_kind:
        raise ValueError("search checkpoint optimizer differs")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("search checkpoint metrics are invalid")
    recorded_metrics = cast(dict[str, object], metrics)
    if recorded_metrics.get("preset") != preset:
        raise ValueError("search checkpoint preset differs")
    if recorded_metrics.get("objective_kind", "improved") != objective_kind:
        raise ValueError("search checkpoint objective differs")


def archive_generation_checkpoint(*, checkpoint_path: Path, artifact_root: Path, generation: int) -> Path:
    """Preserve one immutable full checkpoint for a completed generation."""
    if generation < 1:
        raise ValueError("only completed generations can be archived")
    destination = artifact_root / "generations" / f"generation-{generation:03d}.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    if destination.exists():
        if destination.read_bytes() != checkpoint_bytes:
            raise ValueError(f"generation checkpoint already exists with different contents: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(checkpoint_bytes)
    return destination


def _evaluate_candidates(
    *,
    executor: ProcessPoolExecutor,
    candidates: tuple[Candidate, ...],
    parameter_names: tuple[str, ...],
    base_parameters: ControllerParameters,
    seeds: tuple[int, ...],
    duration_seconds: float,
    preset: SearchPreset,
    objective_kind: ObjectiveKind,
    baseline_distances: dict[int, float],
) -> tuple[CandidateEvaluation, ...]:
    tasks = tuple(
        EvaluationTask(
            candidate=candidate,
            parameter_names=parameter_names,
            base_parameters=base_parameters,
            seeds=seeds,
            duration_seconds=duration_seconds,
        )
        for candidate in candidates
    )
    task_results = tuple(executor.map(_evaluate_task, tasks))
    return tuple(
        CandidateEvaluation(
            candidate=candidate,
            results=results,
            score=_candidate_score(
                preset=preset,
                objective_kind=objective_kind,
                results=results,
                baseline_distances=baseline_distances,
            ),
        )
        for candidate, results in task_results
    )


def _candidate_score(
    *,
    preset: SearchPreset,
    objective_kind: ObjectiveKind,
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    if preset == "minimum":
        return minimum_score(results)
    if objective_kind == "improved-v2":
        return improved_score_v2(results, baseline_distances)
    return improved_score(results, baseline_distances)


def _baseline_distances(
    *,
    executor: ProcessPoolExecutor,
    seeds: tuple[int, ...],
    duration_seconds: float,
) -> dict[int, float]:
    candidate = Candidate(index=-1, values=())
    _, results = next(
        executor.map(
            _evaluate_task,
            (
                EvaluationTask(
                    candidate=candidate,
                    parameter_names=(),
                    base_parameters=MINIMUM_VIABLE_PARAMETERS,
                    seeds=seeds,
                    duration_seconds=duration_seconds,
                ),
            ),
        )
    )
    return {result.seed: result.raw_distance_m for result in results}


def _evaluate_task(task: EvaluationTask) -> tuple[Candidate, tuple[SoloTrialResult, ...]]:
    global _worker_evaluator
    if _worker_evaluator is None:
        _worker_evaluator = SoloEvaluator()
    overrides = dict(zip(task.parameter_names, task.candidate.values, strict=True))
    parameters = replace(task.base_parameters, **overrides)
    results = tuple(
        _worker_evaluator.run_trial(
            controller_factory=lambda: PreviewController(parameters),
            seed=seed,
            duration_seconds=task.duration_seconds,
        )
        for seed in task.seeds
    )
    return task.candidate, results


def preset_configuration(
    preset: str,
    *,
    seed_checkpoint: Path | None = None,
) -> tuple[ControllerParameters, ParameterSpace]:
    """Return the immutable base and ordered parameter space for one preset."""
    if preset == "minimum":
        return MINIMUM_VIABLE_PARAMETERS, minimum_parameter_space()
    if preset == "faster":
        return RACE_FASTER_PARAMETERS, faster_parameter_space()
    if preset == "faster-line":
        return FASTER_LINE_BASE_PARAMETERS, faster_line_parameter_space()
    if preset == "faster-line-v2-probe":
        return FASTER_LINE_V2_BASE_PARAMETERS, faster_line_v2_probe_parameter_space()
    if preset == "faster-line-v3":
        base = FASTER_LINE_V3_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v3_parameter_space(base)
    if preset == "faster-line-v2":
        base = FASTER_LINE_V2_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v2_parameter_space(base)
    raise ValueError(f"unsupported search preset: {preset}")


def _preset_configuration(preset: str) -> tuple[ControllerParameters, ParameterSpace]:
    """Compatibility alias used by the bake tool and existing tests."""
    return preset_configuration(preset)


def _checkpoint_parameter_values(path: Path) -> dict[str, float]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("seed checkpoint is invalid")
    vector = cast(dict[str, object], raw).get("best_parameter_vector")
    if not isinstance(vector, dict):
        raise ValueError("seed checkpoint has no best parameter vector")
    fields = ControllerParameters.__dataclass_fields__
    values: dict[str, float] = {}
    for name, value in cast(dict[str, object], vector).items():
        if name not in fields:
            raise ValueError(f"seed checkpoint parameter is unknown: {name}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
            raise ValueError(f"seed checkpoint parameter is not finite: {name}")
        values[name] = float(value)
    return values


def _checkpoint_best(path: Path, space: ParameterSpace) -> tuple[Candidate, Score]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parameters = raw["best_parameter_vector"]
    metrics = raw["metrics"]
    return (
        Candidate(
            index=int(raw["best_candidate_index"]),
            values=tuple(float(parameters[name]) for name in space.names),
        ),
        tuple(float(value) for value in metrics["best_score"]),
    )


def _checkpoint_metrics(
    *,
    preset: SearchPreset,
    selection_seeds: tuple[int, ...],
    best_score: Score,
    generation_best: CandidateEvaluation,
    promoted_results: tuple[SoloTrialResult, ...],
    duration_seconds: float,
    optimizer_kind: OptimizerKind,
    objective_kind: ObjectiveKind,
    diversity_metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "preset": preset,
        "optimizer_kind": optimizer_kind,
        "objective_kind": objective_kind,
        "duration_seconds": duration_seconds,
        "selection_seeds": list(selection_seeds),
        "best_score": list(best_score),
        "generation_best_score": list(generation_best.score),
        "generation_best_results": [result.to_dict() for result in generation_best.results],
        "new_overall_best_results": [result.to_dict() for result in promoted_results],
        "diversity": diversity_metrics,
    }


def _diversity_metrics(
    *,
    space: ParameterSpace,
    ranked_batch: tuple[CandidateEvaluation, ...],
    ranked_elites: tuple[CandidateEvaluation, ...],
    elite_count: int,
) -> dict[str, object]:
    elite_vectors = tuple(evaluation.candidate.values for evaluation in ranked_elites)
    pairwise: list[float] = []
    for first_index, first in enumerate(elite_vectors):
        for second in elite_vectors[first_index + 1 :]:
            pairwise.append(
                sqrt(
                    sum(
                        ((first_value - second_value) / (spec.maximum - spec.minimum)) ** 2
                        for spec, first_value, second_value in zip(space.specs, first, second, strict=True)
                    )
                )
            )
    per_gene_spread = {
        spec.name: pstdev(vector[index] for vector in elite_vectors) / (spec.maximum - spec.minimum)
        for index, spec in enumerate(space.specs)
    }
    cutoff = ranked_batch[elite_count - 1]
    tier_counts = [0 for _ in cutoff.score]
    tied_at_cutoff = 0
    for evaluation in ranked_batch[elite_count:]:
        for index, (candidate_value, cutoff_value) in enumerate(zip(evaluation.score, cutoff.score, strict=True)):
            if candidate_value != cutoff_value:
                tier_counts[index] += 1
                break
        else:
            tied_at_cutoff += 1
    return {
        "mean_pairwise_bound_normalized_l2": fmean(pairwise) if pairwise else 0.0,
        "elite_spread_by_parameter": per_gene_spread,
        "rejection_count_by_score_tier": tier_counts,
        "rejection_count_tied_at_cutoff": tied_at_cutoff,
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deterministic controller-search command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "preset",
        choices=("minimum", "faster", "faster-line", "faster-line-v2-probe", "faster-line-v2", "faster-line-v3"),
    )
    parser.add_argument("--optimizer", choices=("cem", "ga"), default="cem")
    parser.add_argument("--objective", choices=("improved", "improved-v2"), default="improved")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--optimizer-seed", type=int, default=590_112)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--seconds", type=float, default=SOLO_TRIAL_DEFAULT_SECONDS)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    preset = cast(SearchPreset, args.preset)
    optimizer_kind = cast(OptimizerKind, args.optimizer)
    objective_kind = cast(ObjectiveKind, args.objective)
    seed_checkpoint = cast(Path | None, args.seed_checkpoint)
    # `--seed-checkpoint` is optional: it only supplies the two line-frame spring
    # compensations that the 17-gene space fixes rather than searches.  The v2
    # base already carries 1.0 for both, within 2% of what the probe converged
    # to, so an unseeded run starts from the incumbent instead of inheriting a
    # probe's other genes.  Never seed from a probe that ran against a different
    # line geometry; its collapsed line genes become generation 0's centre.
    default_artifact_name = preset if optimizer_kind == "cem" else f"{preset}-ga"
    artifact_root = (
        DEFAULT_ARTIFACT_ROOT / default_artifact_name
        if args.artifact_root is None
        else cast(Path, args.artifact_root).resolve()
    )
    common_config = {
        "population_size": int(args.population),
        "elite_count": int(args.elites),
        "generations": int(args.generations),
        "optimizer_seed": int(args.optimizer_seed),
    }
    config: CEMConfig | GAConfig = GAConfig(**common_config) if optimizer_kind == "ga" else CEMConfig(**common_config)
    run_search(
        preset=preset,
        artifact_root=artifact_root,
        config=config,
        optimizer_kind=optimizer_kind,
        objective_kind=objective_kind,
        seed_checkpoint=seed_checkpoint,
        duration_seconds=float(args.seconds),
        worker_count=int(args.workers),
    )


if __name__ == "__main__":
    main()
