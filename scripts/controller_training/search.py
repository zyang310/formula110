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
from scripts.controller_training.seeds import SeedManifest, generate_seed_manifest, write_seed_manifest

from controllers.minimum_viable import MINIMUM_VIABLE_PARAMETERS
from controllers.preview_controller import ControllerParameters, PreviewController
from controllers.race_faster import RACE_FASTER_PARAMETERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "controller-search"
DEFAULT_WORKER_COUNT = max(1, (os.cpu_count() or 1) - 1)
# Each worker holds a persistent headless Panda3D scene and builds a fresh Bullet
# world per trial, and that accumulates: across a long run the workers grow until
# the machine swaps, and generation time climbs with it.  A v2 run went 56.7 s to
# 77.8 s over nine generations and dropped straight back to 55.9 s when the
# process was restarted; a v3 run reached 557 s per generation with swap
# exhausted.  The worker rebuilds its evaluator every this many trials, which
# releases the accumulated scene without changing any result: a trial is a pure
# function of its parameters and seed, and the optimizer's RNG lives in the
# parent process.  Set to 0 to disable.
#
# Do NOT implement this with `ProcessPoolExecutor(max_tasks_per_child=...)`.
# That was tried and hung the pool: the workers exited and were never replaced,
# leaving the parent blocked forever on futures at 0% CPU.
DEFAULT_EVALUATOR_RECYCLE_TRIALS = 600
ROTATING_SEED_COUNT = 6
SearchPreset = Literal[
    "minimum",
    "faster",
    "faster-line",
    "faster-line-v2-probe",
    "faster-line-v2",
    "faster-line-v3",
    "faster-line-v4",
    "faster-line-v5",
    "faster-line-v6",
    "faster-line-v7",
    "faster-line-v8",
    "faster-line-v9",
    "faster-line-v10",
    "faster-line-v11",
    "faster-line-v12",
    "faster-line-v13",
    "faster-line-v14",
    "faster-line-v15",
    "faster-line-v16",
    "faster-line-v17",
    "faster-line-v18",
    "faster-line-v19",
    "faster-line-v20",
    "faster-line-v21",
    "faster-line-v22",
    "faster-line-v23",
    "faster-line-v24",
    "faster-line-v25",
]
OptimizerKind = Literal["cem", "ga"]
ObjectiveKind = Literal[
    "improved",
    "improved-v2",
    "lap-time",
    "lap-time-v2",
    "lap-time-v3",
    "lap-time-v4",
    "lap-time-v5",
    "lap-time-v6",
    "lap-time-v7",
    "lap-time-v8",
    "lap-time-v9",
]
Score = tuple[float, ...]

# Per-trial safety budget for `improved_score`. Damage is capped well below the 1.0
# elimination threshold; the distance penalties keep a gradient inside the budget.
INCIDENT_BUDGET_DAMAGE = 0.25
INCIDENT_BUDGET_CONTACT_S = 1.50
# V25 intentionally allows a brushed wall to compete when it buys real lap
# time, while still rejecting heavy damage or prolonged barrier riding.
RELAXED_INCIDENT_BUDGET_DAMAGE = 0.50
RELAXED_INCIDENT_BUDGET_CONTACT_S = 2.00
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


# The v3 generation-60 vector, which is what `race_faster` currently carries, plus
# the asymmetric target release.  Nine of v3's sixteen genes finished on a bound,
# so v4 again moves bounds rather than searching longer.
FASTER_LINE_V4_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    # Start where v3's symmetric behaviour was, so generation 0 reproduces it.
    line_target_release_per_tick=RACE_FASTER_PARAMETERS.line_target_slew_per_tick,
)


# The v4 generation-57 vector was the v5 campaign's starting point; `race_faster`
# now carries the completed v5 winner.  v5 kept v4's line ceilings rather than
# widening again: at 0.95 the car reached 3.77 m of offset on seed 110 with
# 0.29 m of wall clearance, 21 AVOID ticks, and the first wall contact any
# promoted vector had shown.
#
# Two obvious remedies were measured on seed 110 and both are worse, so neither
# is applied here.  Raising `line_clearance_m` destabilises the target rather
# than protecting it: at 2.0 the retraction fires whenever the car runs wide and
# the trial collapses to 234 m with 728 AVOID ticks and 0.38 damage.  Lowering
# the clamp does not reduce peak offset either, which stays near 3.8-4.0 m at
# every setting from 0.95 down to 0.75, while distance falls and AVOID rises.
# The wide excursions therefore are not the racing line, and v4's settings are
# the best measured point on distance, lap time, AVOID count, and clearance
# alike.  Finding what actually drives them is open work, not a v5 bound.
FASTER_LINE_V5_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
)


# The completed v5 generation-100 vector, which `race_faster` carries.  The v5
# GA already inflated mutation_scale to its 0.35 ceiling and retained useful
# diversity, yet gained only 1.56% in 100 generations.  v6 therefore changes
# the shape of the search instead of making mutations still larger: four
# previously fixed steering-dynamics values become genes, and the non-geometric
# bounds v5 pressed against are reopened.  The measured-safe physical line
# clamp stays fixed at 0.95.
FASTER_LINE_V6_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    maximum_racing_line_offset_ratio=0.95,
)


# V6 could not beat this baked v5 vector in ten generations despite searching
# four additional steering dynamics.  A seed-110 trace identified the inherited
# wall-speed policy as the actual lap-time bottleneck: 23 AVOID ticks cap speed
# at 3.4 m/s and another 77 ticks invoke side slowdown.  V7 keeps the proven
# steering and line fixed and searches the speed guards directly under a robust
# lap-time objective with the same hard survival and incident tiers.
FASTER_LINE_V7_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    maximum_racing_line_offset_ratio=0.95,
)


# V8 is seeded explicitly from v7's generation-27 archive.  It keeps v7's
# focused wall/speed hypothesis, removes two speed-reduction genes that stayed
# at zero, and moves the bounds that the real v7 winner pressed.  The v2 lap
# objective quantizes times before lexicographic comparison so sub-femtosecond
# floating noise cannot outrank a materially better median lap.
FASTER_LINE_V8_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    maximum_racing_line_offset_ratio=0.95,
)


# V8 reduced the measured training lap but exploited the objective by hitting a
# wall, recovering, and setting one fast later lap.  V9 restarts from v7's clean
# generation-27 winner, retains the refined speed-policy box, and ranks three-lap
# consistency plus completely clean trials ahead of best-lap time.
FASTER_LINE_V9_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    maximum_racing_line_offset_ratio=0.95,
)


# V9 generation 26 is robustly fast on the 28 training seeds and the second
# official seed, but seed 110 enters its first turn before the racing-line state
# has settled, hits the wall, and spends more than a second recovering.  A
# launch-only 19 m/s cap through 3.5 s removed that correction in an exact
# trace: first lap 9.967 -> 8.667 s, distance 645.01 -> 679.47 m, and contact
# 0.150 -> 0.000 s, while later laps remained uncapped.  V10 starts from that
# measured point and tunes it jointly with the wall/speed dynamics that remain
# active in the trace.
FASTER_LINE_V10_BASE_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    pose_invariant_racing_line=True,
    pose_invariant_speed_curvature=True,
    maximum_racing_line_offset_ratio=0.95,
    startup_speed_cap_mps=19.0,
    startup_speed_cap_seconds=3.5,
)

# V10 demonstrated that ranking every repeated-lap statistic before launch time
# can prefer an 18 m/s four-second cap: clean and one tick faster later, but 1.3
# seconds slower on lap one and 10 m worse over the trial.  V11 keeps the same
# tested behavior and parameter box but changes the ranking to robust estimated
# three-lap time before decomposing repeated and first-lap metrics.
FASTER_LINE_V11_BASE_PARAMETERS = FASTER_LINE_V10_BASE_PARAMETERS

# V11's first generation found a second decimal-ranking artifact: two physical
# three-lap totals of exactly 1,467 simulator ticks became 24.449999 and
# 24.450001 after rounding their constituent seconds.  V12 represents every lap
# as its integer 60 Hz tick count before aggregation, so physically equal times
# tie exactly and the next distribution component decides.
FASTER_LINE_V12_BASE_PARAMETERS = FASTER_LINE_V11_BASE_PARAMETERS

# V12 exposed that seed checkpoints restored only searched values and silently
# discarded their fixed context.  V13 is the first clean campaign after the
# loader was corrected to merge context before the vector.
FASTER_LINE_V13_BASE_PARAMETERS = FASTER_LINE_V12_BASE_PARAMETERS

# The v13 generation-22 winner is clean across all 30 training/official seeds,
# but a seed-110 trace showed it coasting through the uniquely long sweeper.
# A measured local bonus improved the full 30-seed three-lap distribution while
# remaining completely clean.  V14 searches only that activation envelope; its
# seed checkpoint supplies the complete v13 controller around these initials.
FASTER_LINE_V14_BASE_PARAMETERS = replace(
    FASTER_LINE_V13_BASE_PARAMETERS,
    sweeper_minimum_duration_s=1.7,
    sweeper_speed_hold_seconds=0.9,
    sweeper_target_speed_bonus_mps=1.5,
)

# V14's duration gate can identify the long sweeper only after 2.15 seconds of
# coasting through its 2.38-second span.  Its entry is visible much earlier as a
# distinctive far-dominant sign change: |far curvature| is about 0.128 while the
# tighter transition reaches about 0.17.  V15 searches that pose-invariant shape
# band, its hold, and its bonus.  A zero initial bonus makes generation 0 member
# zero reproduce the promoted v14 vector exactly.
FASTER_LINE_V15_BASE_PARAMETERS = replace(
    FASTER_LINE_V14_BASE_PARAMETERS,
    sweeper_preview_minimum_far_curvature=0.10,
    sweeper_preview_maximum_far_curvature=0.14,
    sweeper_preview_speed_hold_seconds=2.30,
    sweeper_preview_target_speed_bonus_mps=0.0,
)

# V15 improved repeated-lap pace, but its inherited launch cap still binds for
# 94-123 ticks on the official traces.  The older first-corner search pinned the
# duration to its 2.5-second floor.  A direct sweep found 2.0 seconds at the
# existing 22.6 m/s cap clean and 5-6 ticks faster on both official first laps;
# shorter durations or higher caps caused seed-110 contact.  V16 searches only
# that narrow measured boundary from the v15 winner.
FASTER_LINE_V16_BASE_PARAMETERS = FASTER_LINE_V15_BASE_PARAMETERS

# V16 optimized the three-lap objective, which weights repeated pace twice.
# The user explicitly wants both the first and best lap minimized.  V17 reopens
# the four preview and two launch genes jointly under an equal-weight exact-tick
# first+best objective, seeded from the balanced v16 winner.
FASTER_LINE_V17_BASE_PARAMETERS = FASTER_LINE_V16_BASE_PARAMETERS

# V17 showed that joint mutation of the existing preview/launch genes was flat.
# V18 adds a new default-off structural lever: accelerate as the pose-invariant
# near curvature unwinds relative to far curvature, rather than waiting for the
# whole speed scalar to fall back toward straight-line pace.
FASTER_LINE_V18_BASE_PARAMETERS = FASTER_LINE_V17_BASE_PARAMETERS

# V13's elite spread pinned two genes hard against their own box: the winning
# corner target sat exactly on the 14.0 m/s floor and the winning front stop
# exactly on the 1.60 m ceiling.  V16 already showed what that signature is
# worth - reopening v13's pinned 2.5 s launch floor bought twelve ticks - so
# V19 reopens the remaining two binding bounds and nothing else.
FASTER_LINE_V19_BASE_PARAMETERS = FASTER_LINE_V18_BASE_PARAMETERS

# Worst first-lap time sat frozen at 513 ticks across v16, v17, v18, and v19:
# only the launch genes have ever moved it, and v16's own box is now partly
# pinned, with four of twelve elites on its 1.85 s floor and the winner within
# a tenth of its 22.9 m/s ceiling.  V20 reopens that box on both sides.
FASTER_LINE_V20_BASE_PARAMETERS = FASTER_LINE_V19_BASE_PARAMETERS

# D-040 rejected 26-27 m/s straight targets because they caused wall contact,
# but that was measured against the *old* corner approach: a 14.0 m/s corner
# target braking to a 1.60 m front stop.  V19 replaced both (13.744 m/s and
# 2.037 m), so the car now arrives at corners slower and braking earlier, and
# the straight-speed ceiling that rejection established no longer applies to
# this controller.  V21 retests it, with the brake ramp free to start earlier.
FASTER_LINE_V21_BASE_PARAMETERS = FASTER_LINE_V20_BASE_PARAMETERS

# V21 exposed that ranking the first+best sum ahead of its parts lets the GA
# bank a first-lap tick by spending a best-lap one, since the sum ties and
# first lap outranks best lap.  `lap_time_score_v8` leads with best lap
# instead.  V22 re-searches the whole speed profile under it: the four genes
# that set straight pace, corner pace, and both ends of the brake ramp, all
# with the bounds v19 and v21 reopened.
FASTER_LINE_V22_BASE_PARAMETERS = FASTER_LINE_V21_BASE_PARAMETERS

# V22 exhausted the speed profile and v20 the launch, so what is left is the
# line's timing rather than its magnitude.  Magnitude is genuinely capped:
# `center_offset_cap_m` is the 3.3 m corridor half-width, so the 0.95 clamp
# already requests 3.135 m and 1.0 is the wall.  Timing is not capped, and
# two of its genes sit exactly on a bound - `line_turn_sensitivity` on the
# 0.002 floor and `line_target_release_per_tick` on the 0.25 ceiling.
FASTER_LINE_V23_BASE_PARAMETERS = FASTER_LINE_V22_BASE_PARAMETERS

# The user identified the opening hairpin as a different control regime: a
# short reverse-throttle rotation impulse, followed by neutral and a brief
# straightening phase, can trade a little longitudinal speed for much earlier
# yaw without the six-tick emergency AVOID brake seen in the seed-110 trace.
# V24 adds that default-off structural lever and jointly reopens the two *real*
# pace limits: the launch target and straight target.  `speed_cap_mps` is only
# feature normalization and is deliberately not searched.
FASTER_LINE_V24_BASE_PARAMETERS = replace(
    FASTER_LINE_V23_BASE_PARAMETERS,
    startup_drift_window_seconds=3.25,
    startup_drift_trigger_front_m=8.5,
    startup_drift_minimum_speed_mps=14.0,
    startup_drift_minimum_steer=0.30,
    startup_drift_pulse_seconds=0.08,
    startup_drift_steer_gain=1.35,
)

# V24 mixed a proven seed-110 opening pulse with two already-exhausted global
# speed genes and stayed flat for ten generations under a perfectly-clean gate.
# V25 raises speed only after a sustained pose-invariant straight, targeting the
# long final corridor, and evaluates it with the bounded-damage v9 objective.
FASTER_LINE_V25_BASE_PARAMETERS = replace(
    FASTER_LINE_V24_BASE_PARAMETERS,
    long_straight_minimum_duration_s=0.35,
    long_straight_maximum_local_curvature=0.012,
    long_straight_speed_bonus_seconds=0.60,
    long_straight_target_speed_bonus_mps=0.0,
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
    evaluator_recycle_trials: int = DEFAULT_EVALUATOR_RECYCLE_TRIALS


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One candidate's results and lexicographic score."""

    candidate: Candidate
    results: tuple[SoloTrialResult, ...]
    score: Score


_worker_evaluator: SoloEvaluator | None = None
_worker_trial_count: int = 0


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
            _spec("heading_steer_gain", 0.84, 1.16, _bounded(base.heading_steer_gain, 0.84, 1.16), 0.065, 0.003),
            _spec("center_steer_gain", 0.40, 0.70, _bounded(base.center_steer_gain, 0.40, 0.70), 0.060, 0.003),
            _spec("yaw_damping_gain", 0.10, 0.28, _bounded(base.yaw_damping_gain, 0.10, 0.28), 0.040, 0.002),
            _spec(
                "racing_line_offset_ratio", 0.0, 0.24, _bounded(base.racing_line_offset_ratio, 0.0, 0.24), 0.050, 0.003
            ),
            _spec(
                "curvature_heading_degrees",
                44.0,
                76.0,
                _bounded(base.curvature_heading_degrees, 44.0, 76.0),
                6.0,
                0.25,
            ),
            _spec(
                "curvature_lateral_ratio", 0.54, 0.95, _bounded(base.curvature_lateral_ratio, 0.54, 0.95), 0.080, 0.004
            ),
            _spec(
                "straight_target_speed_mps",
                13.0,
                17.5,
                _bounded(base.straight_target_speed_mps, 13.0, 17.5),
                0.80,
                0.04,
            ),
            _spec("corner_target_speed_mps", 7.5, 11.5, _bounded(base.corner_target_speed_mps, 7.5, 11.5), 0.70, 0.04),
            _spec(
                "steering_speed_reduction",
                0.06,
                0.26,
                _bounded(base.steering_speed_reduction, 0.06, 0.26),
                0.045,
                0.003,
            ),
            _spec("yaw_speed_reduction", 0.05, 0.24, _bounded(base.yaw_speed_reduction, 0.05, 0.24), 0.040, 0.003),
            _spec("front_brake_start_m", 4.5, 7.5, _bounded(base.front_brake_start_m, 4.5, 7.5), 0.60, 0.04),
            _spec("side_slow_start_m", 0.75, 1.50, _bounded(base.side_slow_start_m, 0.75, 1.50), 0.14, 0.01),
            _spec("side_speed_floor", 0.50, 0.90, _bounded(base.side_speed_floor, 0.50, 0.90), 0.08, 0.005),
            _spec("brake_gain", 0.17, 0.42, _bounded(base.brake_gain, 0.17, 0.42), 0.050, 0.003),
        )
    )


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _spec(
    name: str,
    minimum: float,
    maximum: float,
    initial: float,
    initial_deviation: float,
    minimum_deviation: float,
) -> ParameterSpec:
    """Build a spec whose initial is pulled inside its own bounds.

    Every preset below derives its base from whatever is currently baked into
    `race_faster`, while `ParameterSpec` rejects an out-of-bounds initial.  A
    later bake outside a closed preset's bounds would therefore make that closed
    search unconstructible.  The recorded bounds are asserted by tests and must
    not move, and resuming reads the checkpoint rather than these initials, so
    clamping the initial changes no history.
    """
    return ParameterSpec(
        name,
        minimum,
        maximum,
        _bounded(initial, minimum, maximum),
        initial_deviation,
        minimum_deviation,
    )


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
            _spec("straight_target_speed_mps", 18.0, 26.0, 21.0, 1.6, 0.08),
            _spec("corner_target_speed_mps", 11.0, 17.0, 13.5, 1.2, 0.06),
            _spec("throttle_gain", 0.18, 0.90, 0.45, 0.15, 0.008),
            _spec("front_brake_start_m", 4.0, 14.0, 8.0, 1.8, 0.06),
            _spec("steering_speed_reduction", 0.0, 0.35, 0.10, 0.07, 0.004),
            _spec("yaw_speed_reduction", 0.0, 0.30, 0.08, 0.06, 0.004),
            _spec("curvature_heading_degrees", 45.0, 90.0, 64.0, 8.0, 0.30),
            _spec("curvature_lateral_ratio", 0.60, 1.40, 0.95, 0.14, 0.007),
            _spec("heading_steer_gain", 0.75, 1.30, 0.94, 0.10, 0.005),
            _spec("center_steer_gain", 0.10, 0.55, 0.30, 0.09, 0.004),
            _spec("wall_balance_gain", 0.0, 0.45, 0.20, 0.10, 0.005),
            _spec("steer_slew_per_tick", 0.06, 0.20, 0.10, 0.025, 0.0015),
            _spec("racing_line_offset_ratio", 0.0, 0.45, 0.10, 0.10, 0.006),
            _spec("racing_line_entry_offset_ratio", 0.0, 0.45, 0.10, 0.10, 0.006),
            _spec("racing_line_exit_offset_ratio", 0.0, 0.45, 0.10, 0.10, 0.006),
        )
    )


def faster_line_v2_probe_parameter_space(
    base: ControllerParameters = FASTER_LINE_V2_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 19-gene falsification space for the pose-invariant line.

    Initials are clamped into the recorded bounds.  This preset's base derives
    from whatever is currently baked into `race_faster`, so without the clamp a
    later bake outside these bounds makes a closed search unconstructible; the
    bounds themselves are asserted by tests and must not move.  Resuming reads
    the checkpoint rather than these initials, so clamping changes no history.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 18.0, 26.0, base.straight_target_speed_mps, 1.6, 0.08),
            _spec("corner_target_speed_mps", 13.0, 24.0, base.corner_target_speed_mps, 2.0, 0.10),
            _spec("throttle_gain", 0.30, 2.00, base.throttle_gain, 0.30, 0.015),
            _spec("front_brake_start_m", 4.0, 14.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            _spec("curvature_lateral_ratio", 0.20, 1.00, 0.45, 0.16, 0.01),
            _spec("heading_steer_gain", 0.60, 1.30, base.heading_steer_gain, 0.12, 0.006),
            _spec("center_steer_gain", 0.10, 0.90, base.center_steer_gain, 0.16, 0.008),
            _spec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            _spec("racing_line_offset_ratio", 0.0, 0.65, base.racing_line_offset_ratio, 0.12, 0.008),
            _spec("racing_line_entry_offset_ratio", 0.0, 0.65, base.racing_line_entry_offset_ratio, 0.12, 0.008),
            _spec("racing_line_exit_offset_ratio", 0.0, 0.65, base.racing_line_exit_offset_ratio, 0.12, 0.008),
            _spec("curvature_offset_compensation", 0.0, 1.0, base.curvature_offset_compensation, 0.20, 0.01),
            _spec("curvature_heading_compensation", 0.0, 1.0, base.curvature_heading_compensation, 0.20, 0.01),
            _spec("preview_line_compensation", 0.0, 1.0, base.preview_line_compensation, 0.20, 0.01),
            _spec("wall_balance_line_compensation", 0.0, 1.0, base.wall_balance_line_compensation, 0.20, 0.01),
            # The phase signal is a local curvature in 1/m, which peaks at about
            # 0.16 on this track; the old [0.03, 0.40] box spent over half its
            # width on values that switch the racing line off entirely.
            _spec("line_turn_sensitivity", 0.010, 0.150, base.line_turn_sensitivity, 0.030, 0.0015),
            _spec("line_target_slew_per_tick", 0.005, 0.10, base.line_target_slew_per_tick, 0.02, 0.001),
            _spec("line_clearance_m", 0.0, 2.5, base.line_clearance_m, 0.50, 0.025),
        )
    )


def faster_line_v5_parameter_space(
    base: ControllerParameters = FASTER_LINE_V5_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 17-gene space that restores wall margin and re-seeks pace.

    The line clamp and the three line ratios deliberately keep v4's ceilings.
    v4 reached 3.77 m of offset, which puts the body edge at 4.40 m against a
    barrier at 4.70 m, so widening further trades distance for contact rather
    than for lap time.  Pace is sought in the speed and steering genes instead.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 18.0, 32.0, base.straight_target_speed_mps, 1.8, 0.09),
            _spec("corner_target_speed_mps", 13.0, 30.0, base.corner_target_speed_mps, 2.0, 0.10),
            # v4 finished at 3.68 of a 4.0 ceiling.
            _spec("throttle_gain", 0.30, 6.00, base.throttle_gain, 0.50, 0.025),
            _spec("front_brake_start_m", 4.0, 16.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            # v4 pinned this at the 0.60 ceiling, itself raised from v3's 0.30.
            _spec("curvature_lateral_ratio", 0.02, 1.20, base.curvature_lateral_ratio, 0.15, 0.008),
            # v4 pinned this on the 0.15 floor, itself lowered from v3's 0.40.
            _spec("heading_steer_gain", 0.05, 1.30, base.heading_steer_gain, 0.14, 0.007),
            _spec("center_steer_gain", 0.0, 0.90, base.center_steer_gain, 0.16, 0.008),
            _spec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            # Held at v4's ceilings on purpose. See the docstring.
            _spec("maximum_racing_line_offset_ratio", 0.65, 0.95, base.maximum_racing_line_offset_ratio, 0.06, 0.003),
            _spec("racing_line_offset_ratio", 0.0, 0.95, base.racing_line_offset_ratio, 0.14, 0.008),
            _spec("racing_line_entry_offset_ratio", 0.0, 0.95, base.racing_line_entry_offset_ratio, 0.14, 0.008),
            _spec("racing_line_exit_offset_ratio", 0.0, 0.95, base.racing_line_exit_offset_ratio, 0.14, 0.008),
            _spec("line_turn_sensitivity", 0.002, 0.150, base.line_turn_sensitivity, 0.030, 0.0015),
            _spec("line_target_slew_per_tick", 0.005, 0.25, base.line_target_slew_per_tick, 0.03, 0.0015),
            _spec("line_target_release_per_tick", 0.0005, 0.25, _release_initial(base), 0.03, 0.0015),
            # Kept wide rather than floored: a seed-110 sweep showed high values
            # oscillate the target and collapse the trial, so the search must
            # stay free to choose a low one.
            _spec("line_clearance_m", 0.0, 3.0, base.line_clearance_m, 0.50, 0.025),
        )
    )


def faster_line_v6_parameter_space(
    base: ControllerParameters = FASTER_LINE_V6_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 20-gene structural-variation space seeded from v5.

    v5 finished with useful population diversity and the GA's mutation scale at
    its ceiling, so simply increasing mutation would spend more trials in the
    same box.  v6 instead searches four steering dynamics that were inherited
    unchanged since v2, while reopening only the non-geometric bounds v5
    pressed.  The maximum racing-line clamp is fixed in the base at 0.95 rather
    than spending another gene rediscovering its ceiling.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 18.0, 32.0, base.straight_target_speed_mps, 1.8, 0.09),
            _spec("corner_target_speed_mps", 13.0, 30.0, base.corner_target_speed_mps, 2.0, 0.10),
            _spec("throttle_gain", 0.30, 6.00, base.throttle_gain, 0.50, 0.025),
            _spec("front_brake_start_m", 4.0, 16.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            # New in v6: these four steering dynamics were fixed throughout v5.
            _spec("yaw_speed_reduction", 0.0, 0.30, base.yaw_speed_reduction, 0.05, 0.003),
            _spec("curvature_heading_degrees", 40.0, 90.0, base.curvature_heading_degrees, 8.0, 0.30),
            # v5 reached 1.175 against a 1.20 ceiling.
            _spec("curvature_lateral_ratio", 0.02, 1.60, base.curvature_lateral_ratio, 0.18, 0.009),
            # v5 pinned this on its 0.05 floor.
            _spec("heading_steer_gain", 0.0, 1.30, base.heading_steer_gain, 0.14, 0.007),
            _spec("center_steer_gain", 0.0, 0.90, base.center_steer_gain, 0.16, 0.008),
            _spec("yaw_damping_gain", 0.0, 0.35, base.yaw_damping_gain, 0.06, 0.003),
            _spec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            _spec("steer_slew_per_tick", 0.04, 0.25, base.steer_slew_per_tick, 0.035, 0.002),
            # Keep every requested line ratio inside the fixed 0.95 clamp.
            _spec("racing_line_offset_ratio", 0.0, 0.95, base.racing_line_offset_ratio, 0.14, 0.008),
            _spec("racing_line_entry_offset_ratio", 0.0, 0.95, base.racing_line_entry_offset_ratio, 0.14, 0.008),
            _spec("racing_line_exit_offset_ratio", 0.0, 0.95, base.racing_line_exit_offset_ratio, 0.14, 0.008),
            # v5 pinned the sensitivity floor and the release ceiling.
            _spec("line_turn_sensitivity", 0.0005, 0.150, base.line_turn_sensitivity, 0.020, 0.001),
            _spec("line_target_slew_per_tick", 0.005, 0.25, base.line_target_slew_per_tick, 0.03, 0.0015),
            _spec("line_target_release_per_tick", 0.0005, 0.60, _release_initial(base), 0.06, 0.003),
            # A seed-110 sweep showed that flooring this destabilises the target.
            _spec("line_clearance_m", 0.0, 3.0, base.line_clearance_m, 0.50, 0.025),
        )
    )


def faster_line_v7_parameter_space(
    base: ControllerParameters = FASTER_LINE_V7_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the focused 16-gene wall/speed space seeded from v5.

    The v5 trace is almost never steering-saturated or slew-limited, while wall
    guards impose large speed cuts.  Keep the v5 racing line and steering gains
    fixed, then search the speed schedule and the exact wall thresholds that
    gate it.  Bounds include the clean seed-110 diagnostic improvements without
    admitting the aggressive side-slowdown setting that failed to complete a
    lap.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 22.0, 36.0, base.straight_target_speed_mps, 2.0, 0.10),
            _spec("corner_target_speed_mps", 16.0, 34.0, base.corner_target_speed_mps, 2.5, 0.12),
            _spec("throttle_gain", 1.0, 8.0, base.throttle_gain, 0.60, 0.03),
            _spec("front_brake_start_m", 4.0, 16.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("front_stop_m", 0.5, 2.5, base.front_stop_m, 0.30, 0.015),
            _spec("steering_speed_reduction", 0.0, 0.10, base.steering_speed_reduction, 0.02, 0.001),
            _spec("yaw_speed_reduction", 0.0, 0.15, base.yaw_speed_reduction, 0.03, 0.0015),
            _spec("curvature_heading_degrees", 45.0, 100.0, base.curvature_heading_degrees, 8.0, 0.30),
            _spec("curvature_lateral_ratio", 0.60, 2.0, base.curvature_lateral_ratio, 0.18, 0.009),
            # The failed 0.8/0.8 side-speed probe sits outside this conservative
            # joint box on the floor parameter.
            _spec("side_slow_start_m", 0.70, 1.50, base.side_slow_start_m, 0.12, 0.006),
            _spec("side_speed_floor", 0.40, 0.72, base.side_speed_floor, 0.05, 0.0025),
            _spec("avoid_front_wall_m", 2.0, 3.5, base.avoid_front_wall_m, 0.24, 0.012),
            _spec("avoid_diagonal_wall_m", 0.80, 1.70, base.avoid_diagonal_wall_m, 0.14, 0.007),
            _spec("avoid_side_wall_m", 0.65, 1.05, base.avoid_side_wall_m, 0.07, 0.0035),
            _spec("avoid_speed_mps", 3.0, 12.0, base.avoid_speed_mps, 1.2, 0.06),
            _spec("avoid_steer_gain", 0.60, 1.40, base.avoid_steer_gain, 0.12, 0.006),
        )
    )


def faster_line_v8_parameter_space(
    base: ControllerParameters = FASTER_LINE_V8_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 14-gene v7 refinement space with its pressed bounds moved."""
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 22.0, 32.0, base.straight_target_speed_mps, 1.6, 0.08),
            _spec("corner_target_speed_mps", 15.0, 28.0, base.corner_target_speed_mps, 2.0, 0.10),
            # V7 generation 27 reached the 8.0 ceiling.
            _spec("throttle_gain", 1.0, 10.0, base.throttle_gain, 0.80, 0.04),
            _spec("front_brake_start_m", 4.0, 16.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("front_stop_m", 0.40, 2.5, base.front_stop_m, 0.30, 0.015),
            _spec("curvature_heading_degrees", 45.0, 120.0, base.curvature_heading_degrees, 10.0, 0.40),
            # V7 repeatedly reached its 2.0 ceiling.
            _spec("curvature_lateral_ratio", 0.60, 3.0, base.curvature_lateral_ratio, 0.24, 0.012),
            _spec("side_slow_start_m", 0.60, 1.50, base.side_slow_start_m, 0.12, 0.006),
            _spec("side_speed_floor", 0.30, 0.72, base.side_speed_floor, 0.06, 0.003),
            _spec("avoid_front_wall_m", 1.80, 3.5, base.avoid_front_wall_m, 0.24, 0.012),
            _spec("avoid_diagonal_wall_m", 0.55, 1.70, base.avoid_diagonal_wall_m, 0.16, 0.008),
            _spec("avoid_side_wall_m", 0.50, 1.05, base.avoid_side_wall_m, 0.08, 0.004),
            _spec("avoid_speed_mps", 2.0, 10.0, base.avoid_speed_mps, 1.0, 0.05),
            _spec("avoid_steer_gain", 0.60, 1.40, base.avoid_steer_gain, 0.12, 0.006),
        )
    )


def faster_line_v9_parameter_space(
    base: ControllerParameters = FASTER_LINE_V9_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return v8's focused box with unsafe side-threshold excursions removed."""
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 22.0, 32.0, base.straight_target_speed_mps, 1.6, 0.08),
            _spec("corner_target_speed_mps", 15.0, 28.0, base.corner_target_speed_mps, 2.0, 0.10),
            _spec("throttle_gain", 1.0, 12.0, base.throttle_gain, 0.90, 0.045),
            _spec("front_brake_start_m", 4.0, 16.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("front_stop_m", 0.40, 3.0, base.front_stop_m, 0.35, 0.018),
            _spec("curvature_heading_degrees", 45.0, 120.0, base.curvature_heading_degrees, 10.0, 0.40),
            _spec("curvature_lateral_ratio", 0.60, 3.0, base.curvature_lateral_ratio, 0.24, 0.012),
            _spec("side_slow_start_m", 0.60, 1.50, base.side_slow_start_m, 0.12, 0.006),
            _spec("side_speed_floor", 0.30, 0.72, base.side_speed_floor, 0.06, 0.003),
            _spec("avoid_front_wall_m", 1.80, 3.5, base.avoid_front_wall_m, 0.24, 0.012),
            _spec("avoid_diagonal_wall_m", 0.55, 1.70, base.avoid_diagonal_wall_m, 0.16, 0.008),
            # V8's 0.50 floor caused the crash/recovery false optimum. V7's clean
            # training winner used 0.70, so v9 keeps a modest exploratory margin.
            _spec("avoid_side_wall_m", 0.60, 1.05, base.avoid_side_wall_m, 0.07, 0.0035),
            _spec("avoid_speed_mps", 2.0, 12.0, base.avoid_speed_mps, 1.2, 0.06),
            _spec("avoid_steer_gain", 0.60, 1.40, base.avoid_steer_gain, 0.12, 0.006),
        )
    )


def faster_line_v10_parameter_space(
    base: ControllerParameters = FASTER_LINE_V10_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the focused first-corner and robust pace space seeded from v9.

    V9's straight target and front-braking distance converged with almost no
    elite spread, so v10 fixes them and spends its variation on the launch-only
    cap, the steering-speed/yaw dynamics implicated by the seed-110 trace, and
    the wall/corner genes that continued to move through generation 26.
    """
    return ParameterSpace(
        (
            _spec("startup_speed_cap_mps", 16.0, 24.0, base.startup_speed_cap_mps, 1.0, 0.05),
            _spec("startup_speed_cap_seconds", 2.5, 5.0, base.startup_speed_cap_seconds, 0.40, 0.02),
            _spec("corner_target_speed_mps", 14.0, 22.0, base.corner_target_speed_mps, 1.2, 0.06),
            _spec("throttle_gain", 2.0, 14.0, base.throttle_gain, 1.0, 0.05),
            _spec("front_stop_m", 0.40, 1.60, base.front_stop_m, 0.18, 0.009),
            _spec("steering_speed_reduction", 0.0, 0.05, base.steering_speed_reduction, 0.010, 0.0005),
            _spec("yaw_damping_gain", 0.05, 0.16, base.yaw_damping_gain, 0.018, 0.0009),
            _spec("curvature_heading_degrees", 70.0, 125.0, base.curvature_heading_degrees, 8.0, 0.30),
            _spec("curvature_lateral_ratio", 1.50, 3.50, base.curvature_lateral_ratio, 0.25, 0.012),
            _spec("side_slow_start_m", 0.55, 1.10, base.side_slow_start_m, 0.08, 0.004),
            _spec("side_speed_floor", 0.40, 0.76, base.side_speed_floor, 0.05, 0.0025),
            _spec("avoid_front_wall_m", 2.50, 3.80, base.avoid_front_wall_m, 0.20, 0.010),
            _spec("avoid_diagonal_wall_m", 0.55, 1.40, base.avoid_diagonal_wall_m, 0.12, 0.006),
            # Keep the v9 safety floor; v8's 0.50 excursion was the rejected
            # crash/recovery optimum.
            _spec("avoid_side_wall_m", 0.60, 0.85, base.avoid_side_wall_m, 0.04, 0.002),
            _spec("avoid_speed_mps", 2.0, 7.0, base.avoid_speed_mps, 0.70, 0.035),
            _spec("avoid_steer_gain", 0.75, 1.35, base.avoid_steer_gain, 0.09, 0.0045),
        )
    )


def faster_line_v11_parameter_space(
    base: ControllerParameters = FASTER_LINE_V11_BASE_PARAMETERS,
) -> ParameterSpace:
    """Reuse v10's measured first-corner box under race-time ranking."""
    return faster_line_v10_parameter_space(base)


def faster_line_v12_parameter_space(
    base: ControllerParameters = FASTER_LINE_V12_BASE_PARAMETERS,
) -> ParameterSpace:
    """Reuse the first-corner box with exact tick-based race-time ranking."""
    return faster_line_v10_parameter_space(base)


def faster_line_v13_parameter_space(
    base: ControllerParameters = FASTER_LINE_V13_BASE_PARAMETERS,
) -> ParameterSpace:
    """Reuse v12's exact-tick box with complete checkpoint seeding."""
    return faster_line_v10_parameter_space(base)


def faster_line_v14_parameter_space(
    base: ControllerParameters = FASTER_LINE_V14_BASE_PARAMETERS,
) -> ParameterSpace:
    """Tune the measured long-sweeper speed bonus without reopening v13."""
    return ParameterSpace(
        (
            _spec("sweeper_minimum_duration_s", 1.25, 2.20, base.sweeper_minimum_duration_s, 0.16, 0.008),
            _spec("sweeper_speed_hold_seconds", 0.15, 1.20, base.sweeper_speed_hold_seconds, 0.18, 0.009),
            _spec(
                "sweeper_target_speed_bonus_mps",
                0.10,
                3.00,
                base.sweeper_target_speed_bonus_mps,
                0.40,
                0.020,
            ),
        )
    )


def faster_line_v15_parameter_space(
    base: ControllerParameters = FASTER_LINE_V15_BASE_PARAMETERS,
) -> ParameterSpace:
    """Tune an entry-preview bonus for the broad sweeper seen in v14 traces."""
    return ParameterSpace(
        (
            _spec(
                "sweeper_preview_minimum_far_curvature",
                0.07,
                0.13,
                base.sweeper_preview_minimum_far_curvature,
                0.012,
                0.0006,
            ),
            _spec(
                "sweeper_preview_maximum_far_curvature",
                0.11,
                0.18,
                base.sweeper_preview_maximum_far_curvature,
                0.012,
                0.0007,
            ),
            _spec(
                "sweeper_preview_speed_hold_seconds",
                0.50,
                3.00,
                base.sweeper_preview_speed_hold_seconds,
                0.35,
                0.018,
            ),
            _spec(
                "sweeper_preview_target_speed_bonus_mps",
                0.0,
                4.0,
                base.sweeper_preview_target_speed_bonus_mps,
                1.0,
                0.05,
            ),
        )
    )


def faster_line_v16_parameter_space(
    base: ControllerParameters = FASTER_LINE_V16_BASE_PARAMETERS,
) -> ParameterSpace:
    """Tune the launch cap on the measured clean edge below v13's old floor."""
    return ParameterSpace(
        (
            _spec("startup_speed_cap_mps", 21.5, 22.9, base.startup_speed_cap_mps, 0.30, 0.015),
            _spec("startup_speed_cap_seconds", 1.85, 2.50, base.startup_speed_cap_seconds, 0.16, 0.008),
        )
    )


def faster_line_v17_parameter_space(
    base: ControllerParameters = FASTER_LINE_V17_BASE_PARAMETERS,
) -> ParameterSpace:
    """Jointly tune sweeper preview and launch pace for first+best lap time."""
    preview = faster_line_v15_parameter_space(base)
    launch = faster_line_v16_parameter_space(base)
    return ParameterSpace(preview.specs + launch.specs)


def faster_line_v18_parameter_space(
    base: ControllerParameters = FASTER_LINE_V18_BASE_PARAMETERS,
) -> ParameterSpace:
    """Tune the default-off pose-invariant corner-exit speed bonus."""
    return ParameterSpace(
        (
            _spec(
                "corner_exit_target_speed_bonus_mps",
                0.0,
                3.0,
                base.corner_exit_target_speed_bonus_mps,
                0.75,
                0.038,
            ),
        )
    )


def faster_line_v19_parameter_space(
    base: ControllerParameters = FASTER_LINE_V19_BASE_PARAMETERS,
) -> ParameterSpace:
    """Reopen the two bounds v13's elites pinned exactly against their box.

    Both genes shape the same corner approach, so they are searched together:
    the corner target speed is released below v10's 14.0 m/s floor, and the
    front stop distance above its 1.60 m ceiling, back toward the 2.5-3.0 m
    ceilings that v5-v7 used before v10 narrowed the box.
    """
    return ParameterSpace(
        (
            _spec("corner_target_speed_mps", 11.0, 16.0, base.corner_target_speed_mps, 0.60, 0.030),
            _spec("front_stop_m", 1.30, 2.60, base.front_stop_m, 0.16, 0.008),
        )
    )


def faster_line_v20_parameter_space(
    base: ControllerParameters = FASTER_LINE_V20_BASE_PARAMETERS,
) -> ParameterSpace:
    """Reopen v16's launch box on both sides to attack the frozen first lap.

    V16 moved worst first-lap time from 518 to 513 ticks and nothing since has
    moved it at all, so the launch is the only measured first-lap lever.  Its
    own box is now partly pinned: four of twelve elites finished on the 1.85 s
    floor, and the winning cap sits within a tenth of the 22.9 m/s ceiling.
    """
    return ParameterSpace(
        (
            _spec("startup_speed_cap_mps", 22.0, 24.5, base.startup_speed_cap_mps, 0.35, 0.018),
            _spec("startup_speed_cap_seconds", 1.30, 2.10, base.startup_speed_cap_seconds, 0.18, 0.009),
        )
    )


def faster_line_v21_parameter_space(
    base: ControllerParameters = FASTER_LINE_V21_BASE_PARAMETERS,
) -> ParameterSpace:
    """Retest the straight-speed ceiling under v19's slower corner approach.

    The two genes are searched together because they trade directly: a higher
    straight target only survives if the front brake ramp may start further out.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 24.5, 28.5, base.straight_target_speed_mps, 0.55, 0.028),
            _spec("front_brake_start_m", 10.0, 15.5, base.front_brake_start_m, 0.70, 0.035),
        )
    )


def faster_line_v22_parameter_space(
    base: ControllerParameters = FASTER_LINE_V22_BASE_PARAMETERS,
) -> ParameterSpace:
    """Search the whole speed profile jointly, ranked best lap first.

    Straight target, corner target, and the two ends of the front brake ramp are
    one mechanism: what the car carries into a corner and what it can carry out.
    V19 and v21 each moved half of it behind a ranking that could trade best lap
    away, so the pair is re-searched together under `lap_time_score_v8`.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 24.5, 28.5, base.straight_target_speed_mps, 0.55, 0.028),
            _spec("corner_target_speed_mps", 11.0, 16.0, base.corner_target_speed_mps, 0.60, 0.030),
            _spec("front_brake_start_m", 10.0, 15.5, base.front_brake_start_m, 0.70, 0.035),
            _spec("front_stop_m", 1.30, 2.60, base.front_stop_m, 0.16, 0.008),
        )
    )


def faster_line_v23_parameter_space(
    base: ControllerParameters = FASTER_LINE_V23_BASE_PARAMETERS,
) -> ParameterSpace:
    """Reopen the two line-timing genes that finished exactly on a bound.

    `line_turn_sensitivity` saturates `turn_strength`, so its 0.002 floor means
    even the gentlest curvature already commands the full offset; the search
    wanted a sharper response still.  `line_target_release_per_tick` sat on its
    0.25 ceiling, meaning it wanted the target to relax faster than the box
    allowed.  Both control when the line moves, not how far, so neither is
    limited by the corridor width that caps the offset clamp.
    """
    return ParameterSpace(
        (
            _spec("line_turn_sensitivity", 0.0002, 0.010, base.line_turn_sensitivity, 0.0016, 0.00008),
            _spec("line_target_release_per_tick", 0.20, 1.00, _release_initial(base), 0.10, 0.005),
        )
    )


def faster_line_v24_parameter_space(
    base: ControllerParameters = FASTER_LINE_V24_BASE_PARAMETERS,
) -> ParameterSpace:
    """Jointly search a one-shot opening drift and higher real speed targets."""
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 25.0, 34.0, base.straight_target_speed_mps, 1.20, 0.060),
            _spec("startup_speed_cap_mps", 22.5, 27.5, base.startup_speed_cap_mps, 0.70, 0.035),
            _spec("startup_drift_brake", 0.0, 0.90, base.startup_drift_brake, 0.20, 0.010),
            _spec("startup_drift_trigger_front_m", 6.5, 10.5, base.startup_drift_trigger_front_m, 0.65, 0.030),
            _spec("startup_drift_minimum_steer", 0.15, 0.65, base.startup_drift_minimum_steer, 0.10, 0.005),
            _spec("startup_drift_pulse_seconds", 1.0 / 60.0, 0.20, base.startup_drift_pulse_seconds, 0.035, 0.003),
            _spec("startup_drift_steer_gain", 1.0, 2.50, base.startup_drift_steer_gain, 0.28, 0.014),
            _spec("startup_drift_straighten_seconds", 0.0, 0.35, base.startup_drift_straighten_seconds, 0.08, 0.004),
        )
    )


def faster_line_v25_parameter_space(
    base: ControllerParameters = FASTER_LINE_V25_BASE_PARAMETERS,
) -> ParameterSpace:
    """Search the long-corridor boost and the independently proven opening drift."""
    return ParameterSpace(
        (
            _spec("long_straight_minimum_duration_s", 0.10, 0.80, base.long_straight_minimum_duration_s, 0.14, 0.007),
            _spec(
                "long_straight_maximum_local_curvature",
                0.003,
                0.035,
                base.long_straight_maximum_local_curvature,
                0.006,
                0.0003,
            ),
            _spec(
                "long_straight_speed_bonus_seconds",
                0.20,
                1.00,
                base.long_straight_speed_bonus_seconds,
                0.16,
                0.008,
            ),
            _spec(
                "long_straight_target_speed_bonus_mps", 0.0, 6.0, base.long_straight_target_speed_bonus_mps, 1.20, 0.060
            ),
            _spec("startup_drift_brake", 0.0, 0.80, base.startup_drift_brake, 0.18, 0.009),
            _spec("startup_drift_trigger_front_m", 6.5, 10.5, base.startup_drift_trigger_front_m, 0.65, 0.030),
            _spec("startup_drift_minimum_steer", 0.15, 0.65, base.startup_drift_minimum_steer, 0.10, 0.005),
            _spec("startup_drift_pulse_seconds", 1.0 / 60.0, 0.18, base.startup_drift_pulse_seconds, 0.035, 0.003),
            _spec("startup_drift_steer_gain", 1.0, 2.20, base.startup_drift_steer_gain, 0.24, 0.012),
        )
    )


def _release_initial(base: ControllerParameters) -> float:
    """Start the release rate at the outward slew, i.e. v3's symmetric behaviour."""
    release = base.line_target_release_per_tick
    return base.line_target_slew_per_tick if release is None else release


def faster_line_v4_parameter_space(
    base: ControllerParameters = FASTER_LINE_V4_BASE_PARAMETERS,
) -> ParameterSpace:
    """Return the 17-gene space: v3's pinned ceilings raised, plus target release.

    The taut shortest path inside this corridor sits pinned to one edge for 64 m
    at a stretch, which a curvature-instantaneous target cannot express; the
    release rate is what lets the line hold through a low-curvature gap.
    """
    return ParameterSpace(
        (
            _spec("straight_target_speed_mps", 18.0, 30.0, base.straight_target_speed_mps, 1.6, 0.08),
            _spec("corner_target_speed_mps", 13.0, 28.0, base.corner_target_speed_mps, 2.0, 0.10),
            # v3 pinned this at its 2.0 ceiling.  Throttle is a clamped proportional
            # term, so a higher gain only reaches full throttle sooner.
            _spec("throttle_gain", 0.30, 4.00, base.throttle_gain, 0.40, 0.02),
            _spec("front_brake_start_m", 4.0, 14.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            # v3 pinned this at its 0.30 ceiling.
            _spec("curvature_lateral_ratio", 0.02, 0.60, base.curvature_lateral_ratio, 0.08, 0.004),
            # v3 pinned this on its 0.40 floor.
            _spec("heading_steer_gain", 0.15, 1.30, base.heading_steer_gain, 0.14, 0.007),
            _spec("center_steer_gain", 0.0, 0.90, base.center_steer_gain, 0.16, 0.008),
            _spec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            # v3 pinned the clamp and both line ratios at 0.90 with zero elite
            # spread.  At 0.95 the body edge sits at 3.77 m against a barrier at
            # 4.07 m; the clearance retraction and AVOID hold the rest.
            _spec("maximum_racing_line_offset_ratio", 0.65, 0.95, base.maximum_racing_line_offset_ratio, 0.08, 0.004),
            _spec("racing_line_offset_ratio", 0.0, 0.95, base.racing_line_offset_ratio, 0.14, 0.008),
            _spec("racing_line_entry_offset_ratio", 0.0, 0.95, base.racing_line_entry_offset_ratio, 0.14, 0.008),
            _spec("racing_line_exit_offset_ratio", 0.0, 0.95, base.racing_line_exit_offset_ratio, 0.14, 0.008),
            _spec("line_turn_sensitivity", 0.002, 0.150, base.line_turn_sensitivity, 0.030, 0.0015),
            _spec("line_target_slew_per_tick", 0.005, 0.25, base.line_target_slew_per_tick, 0.03, 0.0015),
            # The new gene.  At the top of the box it matches the outward slew,
            # which is v3's symmetric behaviour; at the bottom the line holds.
            _spec("line_target_release_per_tick", 0.0005, 0.25, _release_initial(base), 0.03, 0.0015),
            _spec("line_clearance_m", 0.0, 3.0, base.line_clearance_m, 0.50, 0.025),
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
            _spec("straight_target_speed_mps", 18.0, 28.0, base.straight_target_speed_mps, 1.6, 0.08),
            _spec("corner_target_speed_mps", 13.0, 26.0, base.corner_target_speed_mps, 2.0, 0.10),
            _spec("throttle_gain", 0.30, 2.00, base.throttle_gain, 0.30, 0.015),
            _spec("front_brake_start_m", 4.0, 14.0, base.front_brake_start_m, 1.8, 0.06),
            _spec("steering_speed_reduction", 0.0, 0.20, base.steering_speed_reduction, 0.04, 0.003),
            # The speed scalar is now a local curvature in 1/m peaking near 0.16.
            _spec("curvature_lateral_ratio", 0.02, 0.30, base.curvature_lateral_ratio, 0.06, 0.003),
            _spec("heading_steer_gain", 0.40, 1.30, base.heading_steer_gain, 0.12, 0.006),
            # Floor dropped to zero: the v2 winner sat on the old 0.10 floor, and
            # after the line-frame debias the preview term carries the line.
            _spec("center_steer_gain", 0.0, 0.90, base.center_steer_gain, 0.16, 0.008),
            _spec("wall_balance_gain", 0.0, 0.45, base.wall_balance_gain, 0.10, 0.005),
            # The v2 winner pinned entry against its 0.65 clamp, so the clamp is
            # searched here and the three ratios are bounded by its ceiling.
            _spec("maximum_racing_line_offset_ratio", 0.65, 0.90, base.maximum_racing_line_offset_ratio, 0.08, 0.004),
            _spec("racing_line_offset_ratio", 0.0, 0.90, base.racing_line_offset_ratio, 0.14, 0.008),
            _spec("racing_line_entry_offset_ratio", 0.0, 0.90, base.racing_line_entry_offset_ratio, 0.14, 0.008),
            _spec("racing_line_exit_offset_ratio", 0.0, 0.90, base.racing_line_exit_offset_ratio, 0.14, 0.008),
            # Floor dropped: the v2 winner sat on the old 0.010 floor.
            _spec("line_turn_sensitivity", 0.002, 0.150, base.line_turn_sensitivity, 0.030, 0.0015),
            _spec("line_target_slew_per_tick", 0.005, 0.15, base.line_target_slew_per_tick, 0.02, 0.001),
            _spec("line_clearance_m", 0.0, 3.0, base.line_clearance_m, 0.50, 0.025),
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


def _full_evaluation_seeds(preset: SearchPreset, manifest: SeedManifest) -> tuple[int, ...]:
    """Add the known promotion seeds only for v10's generalization repair."""
    if preset in (
        "faster-line-v10",
        "faster-line-v11",
        "faster-line-v12",
        "faster-line-v13",
        "faster-line-v14",
        "faster-line-v15",
        "faster-line-v16",
        "faster-line-v17",
        "faster-line-v18",
        "faster-line-v19",
        "faster-line-v20",
        "faster-line-v21",
        "faster-line-v22",
        "faster-line-v23",
        "faster-line-v24",
        "faster-line-v25",
    ):
        return manifest.training + manifest.official
    return manifest.training


def _selection_evaluation_seeds(
    preset: SearchPreset,
    manifest: SeedManifest,
    generation: int,
) -> tuple[int, ...]:
    """Keep both official seeds visible during v10's rotating preselection."""
    rotating = rotating_training_seeds(manifest.training, generation)
    if preset in (
        "faster-line-v10",
        "faster-line-v11",
        "faster-line-v12",
        "faster-line-v13",
        "faster-line-v14",
        "faster-line-v15",
        "faster-line-v16",
        "faster-line-v17",
        "faster-line-v18",
        "faster-line-v19",
        "faster-line-v20",
        "faster-line-v21",
        "faster-line-v22",
        "faster-line-v23",
        "faster-line-v24",
        "faster-line-v25",
    ):
        return rotating + manifest.official
    return rotating


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


def lap_time_score(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Minimize robust best-lap time after the existing hard safety tiers.

    Scores are maximized, so lap-time components are negated.  A missing lap is
    already rejected by the lap-count tier; the finite fallback keeps every
    tuple JSON/checkpoint safe and gives partial candidates a deterministic
    ordering.  Penalized distance is only the final tie-breaker.
    """
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    lap_times = tuple(
        result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0
        for result in results
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
        -max(lap_times),
        -percentile(lap_times, 0.90),
        -median(lap_times),
        -fmean(lap_times),
        fmean(penalized_distances),
    )


def lap_time_score_v2(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank lap time like v1 after removing insignificant float jitter."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    lap_times = tuple(
        round(
            result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0,
            6,
        )
        for result in results
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
        -max(lap_times),
        -percentile(lap_times, 0.90),
        -median(lap_times),
        -fmean(lap_times),
        fmean(penalized_distances),
    )


def lap_time_score_v3(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Reject crash-then-sprint policies before minimizing robust lap time."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_times = tuple(
        round(
            result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0,
            6,
        )
        for result in results
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        float(clean_count),
        -max(lap_times),
        -percentile(lap_times, 0.90),
        -median(lap_times),
        -fmean(lap_times),
        fmean(penalized_distances),
    )


def lap_time_score_v4(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Extend v3 with robust first-lap tie-breakers for launch cleanliness."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_times = tuple(
        round(
            result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0,
            6,
        )
        for result in results
    )
    first_lap_times = tuple(
        round(
            result.first_lap_time_seconds
            if result.first_lap_time_seconds is not None
            else result.elapsed_seconds * 2.0,
            6,
        )
        for result in results
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        float(clean_count),
        -max(lap_times),
        -percentile(lap_times, 0.90),
        -median(lap_times),
        -fmean(lap_times),
        -max(first_lap_times),
        -percentile(first_lap_times, 0.90),
        -median(first_lap_times),
        -fmean(first_lap_times),
        fmean(penalized_distances),
    )


def lap_time_score_v5(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank robust estimated three-lap time before its lap components.

    Thirty-second trials normally complete three full laps.  Using first lap
    plus twice the best repeated lap prevents a tiny later-lap gain from buying
    a much slower launch, while the decomposed best/first-lap distributions and
    penalized distance retain deterministic tie-breakers.
    """
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_times = tuple(
        round(
            result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0,
            6,
        )
        for result in results
    )
    first_lap_times = tuple(
        round(
            result.first_lap_time_seconds
            if result.first_lap_time_seconds is not None
            else result.elapsed_seconds * 2.0,
            6,
        )
        for result in results
    )
    three_lap_times = tuple(
        round(first + 2.0 * repeated, 6) for first, repeated in zip(first_lap_times, lap_times, strict=True)
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        float(clean_count),
        -max(three_lap_times),
        -percentile(three_lap_times, 0.90),
        -median(three_lap_times),
        -fmean(three_lap_times),
        -max(lap_times),
        -percentile(lap_times, 0.90),
        -median(lap_times),
        -fmean(lap_times),
        -max(first_lap_times),
        -percentile(first_lap_times, 0.90),
        -median(first_lap_times),
        -fmean(first_lap_times),
        fmean(penalized_distances),
    )


def lap_time_score_v6(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank v5's race-time distribution in exact 60 Hz simulator ticks."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_ticks = tuple(
        round(
            (result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0)
            * 60.0
        )
        for result in results
    )
    first_lap_ticks = tuple(
        round(
            (
                result.first_lap_time_seconds
                if result.first_lap_time_seconds is not None
                else result.elapsed_seconds * 2.0
            )
            * 60.0
        )
        for result in results
    )
    three_lap_ticks = tuple(first + 2 * repeated for first, repeated in zip(first_lap_ticks, lap_ticks, strict=True))
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        float(clean_count),
        -float(max(three_lap_ticks)),
        -percentile(tuple(float(value) for value in three_lap_ticks), 0.90),
        -median(three_lap_ticks),
        -fmean(three_lap_ticks),
        -float(max(lap_ticks)),
        -percentile(tuple(float(value) for value in lap_ticks), 0.90),
        -median(lap_ticks),
        -fmean(lap_ticks),
        -float(max(first_lap_ticks)),
        -percentile(tuple(float(value) for value in first_lap_ticks), 0.90),
        -median(first_lap_ticks),
        -fmean(first_lap_ticks),
        fmean(penalized_distances),
    )


def lap_time_score_v7(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank first and best laps with equal weight in exact simulator ticks."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_ticks = tuple(
        round(
            (result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0)
            * 60.0
        )
        for result in results
    )
    first_lap_ticks = tuple(
        round(
            (
                result.first_lap_time_seconds
                if result.first_lap_time_seconds is not None
                else result.elapsed_seconds * 2.0
            )
            * 60.0
        )
        for result in results
    )
    first_plus_best_ticks = tuple(first + repeated for first, repeated in zip(first_lap_ticks, lap_ticks, strict=True))
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        float(clean_count),
        -float(max(first_plus_best_ticks)),
        -float(max(first_lap_ticks)),
        -float(max(lap_ticks)),
        -percentile(tuple(float(value) for value in first_plus_best_ticks), 0.90),
        -percentile(tuple(float(value) for value in first_lap_ticks), 0.90),
        -percentile(tuple(float(value) for value in lap_ticks), 0.90),
        -median(first_plus_best_ticks),
        -median(first_lap_ticks),
        -median(lap_ticks),
        -fmean(first_plus_best_ticks),
        -fmean(first_lap_ticks),
        -fmean(lap_ticks),
        fmean(penalized_distances),
    )


def lap_time_score_v8(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank best lap ahead of first lap, in exact simulator ticks.

    V7 ranked the first+best sum, then first lap, then best lap.  Because the
    sum was pinned, every tie fell through to first lap, which outranks best
    lap, so the search could bank a first-lap tick by spending a best-lap one:
    v21 generation 11 traded 513+457 for 512+458 and scored it as a win even
    though the sum was identical.  Leading with best lap removes that trade.
    Best lap is also the better target on its own terms: a thirty-second trial
    runs three laps, so a repeated lap counts twice against the first lap once.
    """
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= INCIDENT_BUDGET_DAMAGE and result.wall_contact_seconds <= INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_ticks = tuple(
        round(
            (result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0)
            * 60.0
        )
        for result in results
    )
    first_lap_ticks = tuple(
        round(
            (
                result.first_lap_time_seconds
                if result.first_lap_time_seconds is not None
                else result.elapsed_seconds * 2.0
            )
            * 60.0
        )
        for result in results
    )
    first_plus_best_ticks = tuple(first + repeated for first, repeated in zip(first_lap_ticks, lap_ticks, strict=True))
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        float(clean_count),
        -float(max(lap_ticks)),
        -float(max(first_lap_ticks)),
        -float(max(first_plus_best_ticks)),
        -percentile(tuple(float(value) for value in lap_ticks), 0.90),
        -percentile(tuple(float(value) for value in first_lap_ticks), 0.90),
        -percentile(tuple(float(value) for value in first_plus_best_ticks), 0.90),
        -median(lap_ticks),
        -median(first_lap_ticks),
        -median(first_plus_best_ticks),
        -fmean(lap_ticks),
        -fmean(first_lap_ticks),
        -fmean(first_plus_best_ticks),
        fmean(penalized_distances),
    )


def lap_time_score_v9(
    results: tuple[SoloTrialResult, ...],
    baseline_distances: dict[int, float],
) -> Score:
    """Rank best and first laps before cleanliness within a looser incident cap."""
    del baseline_distances
    if not results:
        raise ValueError("candidate evaluation requires at least one trial")
    within_budget = sum(
        1
        for result in results
        if result.damage <= RELAXED_INCIDENT_BUDGET_DAMAGE
        and result.wall_contact_seconds <= RELAXED_INCIDENT_BUDGET_CONTACT_S
    )
    clean_count = sum(1 for result in results if result.damage == 0.0 and result.wall_contact_seconds == 0.0)
    lap_ticks = tuple(
        round(
            (result.best_lap_time_seconds if result.best_lap_time_seconds is not None else result.elapsed_seconds * 2.0)
            * 60.0
        )
        for result in results
    )
    first_lap_ticks = tuple(
        round(
            (
                result.first_lap_time_seconds
                if result.first_lap_time_seconds is not None
                else result.elapsed_seconds * 2.0
            )
            * 60.0
        )
        for result in results
    )
    first_plus_best_ticks = tuple(first + best for first, best in zip(first_lap_ticks, lap_ticks, strict=True))
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
        float(sum(1 for result in results if result.lap_count >= 3)),
        -float(max(lap_ticks)),
        -percentile(tuple(float(value) for value in lap_ticks), 0.90),
        -median(lap_ticks),
        -fmean(lap_ticks),
        -float(max(first_lap_ticks)),
        -percentile(tuple(float(value) for value in first_lap_ticks), 0.90),
        -median(first_lap_ticks),
        -fmean(first_lap_ticks),
        -float(max(first_plus_best_ticks)),
        -fmean(first_plus_best_ticks),
        float(clean_count),
        -fmean(result.damage for result in results),
        -fmean(result.wall_contact_seconds for result in results),
        fmean(penalized_distances),
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
    evaluator_recycle_trials: int = DEFAULT_EVALUATOR_RECYCLE_TRIALS,
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
        if preset == "minimum" or objective_kind == "improved-v2":
            score_length = 4
        elif objective_kind in ("lap-time", "lap-time-v2"):
            score_length = 8
        elif objective_kind == "lap-time-v3":
            score_length = 10
        elif objective_kind == "lap-time-v4":
            score_length = 14
        elif objective_kind in ("lap-time-v5", "lap-time-v6"):
            score_length = 18
        else:
            score_length = 7
        best_score = tuple(float("-inf") for _ in range(score_length))

    if optimizer.complete:
        return best_candidate

    full_evaluation_seeds = _full_evaluation_seeds(preset, manifest)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        baseline_distances = _baseline_distances(
            executor=executor,
            seeds=full_evaluation_seeds,
            duration_seconds=duration_seconds,
        )
        while not optimizer.complete:
            selection_seeds = _selection_evaluation_seeds(preset, manifest, optimizer.generation)
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
                evaluator_recycle_trials=evaluator_recycle_trials,
            )
            ranked_batch = tuple(sorted(batch_evaluations, key=lambda evaluation: evaluation.score, reverse=True))
            selected = tuple(evaluation.candidate for evaluation in ranked_batch[: config.elite_count])
            full_evaluations = _evaluate_candidates(
                executor=executor,
                candidates=selected,
                parameter_names=space.names,
                base_parameters=base_parameters,
                seeds=full_evaluation_seeds,
                duration_seconds=duration_seconds,
                preset=preset,
                objective_kind=objective_kind,
                baseline_distances=baseline_distances,
                evaluator_recycle_trials=evaluator_recycle_trials,
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
    if preset == "faster-line-v25":
        context = _checkpoint_context("faster-line-v24", parameters)
        # The two v24 global speed genes and its rejected forced-straightening
        # phase stay fixed; v25 searches only local corridor pace and drift.
        context["straight_target_speed_mps"] = parameters.straight_target_speed_mps
        context["startup_speed_cap_mps"] = parameters.startup_speed_cap_mps
        context["startup_drift_straighten_seconds"] = parameters.startup_drift_straighten_seconds
        return context
    if preset == "faster-line-v24":
        context = _checkpoint_context("faster-line-v23", parameters)
        # V24 searches the actual speed targets plus every active drift scalar.
        # Preserve v23's line timing and the fixed opening window/minimum speed.
        context["line_turn_sensitivity"] = parameters.line_turn_sensitivity
        context["line_target_release_per_tick"] = _release_initial(parameters)
        context["startup_drift_window_seconds"] = parameters.startup_drift_window_seconds
        context["startup_drift_minimum_speed_mps"] = parameters.startup_drift_minimum_speed_mps
        del context["straight_target_speed_mps"]
        del context["startup_speed_cap_mps"]
        return context
    if preset == "faster-line-v23":
        # V21 and v22 each removed the genes they searched; v23 searches none of
        # them, so every one has to come back or the run silently reverts it to
        # a base default instead of carrying the seeded winner's value.
        context = _checkpoint_context("faster-line-v22", parameters)
        context["corner_target_speed_mps"] = parameters.corner_target_speed_mps
        context["front_stop_m"] = parameters.front_stop_m
        context["straight_target_speed_mps"] = parameters.straight_target_speed_mps
        context["front_brake_start_m"] = parameters.front_brake_start_m
        del context["line_turn_sensitivity"]
        del context["line_target_release_per_tick"]
        return context
    if preset == "faster-line-v22":
        context = _checkpoint_context("faster-line-v21", parameters)
        del context["corner_target_speed_mps"]
        del context["front_stop_m"]
        return context
    if preset == "faster-line-v21":
        context = _checkpoint_context("faster-line-v20", parameters)
        context["startup_speed_cap_mps"] = parameters.startup_speed_cap_mps
        context["startup_speed_cap_seconds"] = parameters.startup_speed_cap_seconds
        del context["straight_target_speed_mps"]
        del context["front_brake_start_m"]
        return context
    if preset == "faster-line-v20":
        context = _checkpoint_context("faster-line-v19", parameters)
        context["corner_target_speed_mps"] = parameters.corner_target_speed_mps
        context["front_stop_m"] = parameters.front_stop_m
        del context["startup_speed_cap_mps"]
        del context["startup_speed_cap_seconds"]
        return context
    if preset == "faster-line-v19":
        context = _checkpoint_context("faster-line-v18", parameters)
        context["corner_exit_target_speed_bonus_mps"] = parameters.corner_exit_target_speed_bonus_mps
        del context["corner_target_speed_mps"]
        del context["front_stop_m"]
        return context
    if preset == "faster-line-v18":
        context = _checkpoint_context("faster-line-v15", parameters)
        context.update(
            {
                "sweeper_preview_minimum_far_curvature": parameters.sweeper_preview_minimum_far_curvature,
                "sweeper_preview_maximum_far_curvature": parameters.sweeper_preview_maximum_far_curvature,
                "sweeper_preview_speed_hold_seconds": parameters.sweeper_preview_speed_hold_seconds,
                "sweeper_preview_target_speed_bonus_mps": parameters.sweeper_preview_target_speed_bonus_mps,
            }
        )
        return context
    if preset == "faster-line-v17":
        context = _checkpoint_context("faster-line-v16", parameters)
        for name in (
            "sweeper_preview_minimum_far_curvature",
            "sweeper_preview_maximum_far_curvature",
            "sweeper_preview_speed_hold_seconds",
            "sweeper_preview_target_speed_bonus_mps",
        ):
            del context[name]
        return context
    if preset == "faster-line-v16":
        context = _checkpoint_context("faster-line-v15", parameters)
        del context["startup_speed_cap_mps"]
        del context["startup_speed_cap_seconds"]
        context.update(
            {
                "sweeper_preview_minimum_far_curvature": parameters.sweeper_preview_minimum_far_curvature,
                "sweeper_preview_maximum_far_curvature": parameters.sweeper_preview_maximum_far_curvature,
                "sweeper_preview_speed_hold_seconds": parameters.sweeper_preview_speed_hold_seconds,
                "sweeper_preview_target_speed_bonus_mps": parameters.sweeper_preview_target_speed_bonus_mps,
            }
        )
        return context
    if preset == "faster-line-v15":
        return {
            "avoid_diagonal_wall_m": parameters.avoid_diagonal_wall_m,
            "avoid_front_wall_m": parameters.avoid_front_wall_m,
            "avoid_side_wall_m": parameters.avoid_side_wall_m,
            "avoid_speed_mps": parameters.avoid_speed_mps,
            "avoid_steer_gain": parameters.avoid_steer_gain,
            "center_steer_gain": parameters.center_steer_gain,
            "corner_target_speed_mps": parameters.corner_target_speed_mps,
            "curvature_heading_degrees": parameters.curvature_heading_degrees,
            "curvature_lateral_ratio": parameters.curvature_lateral_ratio,
            "front_brake_start_m": parameters.front_brake_start_m,
            "front_stop_m": parameters.front_stop_m,
            "heading_steer_gain": parameters.heading_steer_gain,
            "line_clearance_m": parameters.line_clearance_m,
            "line_target_release_per_tick": _release_initial(parameters),
            "line_target_slew_per_tick": parameters.line_target_slew_per_tick,
            "line_turn_sensitivity": parameters.line_turn_sensitivity,
            "maximum_racing_line_offset_ratio": parameters.maximum_racing_line_offset_ratio,
            "normal_steer_limit": parameters.normal_steer_limit,
            "preview_line_compensation": parameters.preview_line_compensation,
            "racing_line_entry_offset_ratio": parameters.racing_line_entry_offset_ratio,
            "racing_line_exit_offset_ratio": parameters.racing_line_exit_offset_ratio,
            "racing_line_offset_ratio": parameters.racing_line_offset_ratio,
            "side_slow_start_m": parameters.side_slow_start_m,
            "side_speed_floor": parameters.side_speed_floor,
            "startup_speed_cap_mps": parameters.startup_speed_cap_mps,
            "startup_speed_cap_seconds": parameters.startup_speed_cap_seconds,
            "steer_slew_per_tick": parameters.steer_slew_per_tick,
            "steering_speed_reduction": parameters.steering_speed_reduction,
            "straight_target_speed_mps": parameters.straight_target_speed_mps,
            "sweeper_minimum_duration_s": parameters.sweeper_minimum_duration_s,
            "sweeper_speed_hold_seconds": parameters.sweeper_speed_hold_seconds,
            "sweeper_target_speed_bonus_mps": parameters.sweeper_target_speed_bonus_mps,
            "throttle_gain": parameters.throttle_gain,
            "wall_balance_gain": parameters.wall_balance_gain,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
            "yaw_damping_gain": parameters.yaw_damping_gain,
            "yaw_speed_reduction": parameters.yaw_speed_reduction,
        }
    if preset == "faster-line-v14":
        return {
            "avoid_diagonal_wall_m": parameters.avoid_diagonal_wall_m,
            "avoid_front_wall_m": parameters.avoid_front_wall_m,
            "avoid_side_wall_m": parameters.avoid_side_wall_m,
            "avoid_speed_mps": parameters.avoid_speed_mps,
            "avoid_steer_gain": parameters.avoid_steer_gain,
            "center_steer_gain": parameters.center_steer_gain,
            "corner_target_speed_mps": parameters.corner_target_speed_mps,
            "curvature_heading_degrees": parameters.curvature_heading_degrees,
            "curvature_lateral_ratio": parameters.curvature_lateral_ratio,
            "front_brake_start_m": parameters.front_brake_start_m,
            "front_stop_m": parameters.front_stop_m,
            "heading_steer_gain": parameters.heading_steer_gain,
            "line_clearance_m": parameters.line_clearance_m,
            "line_target_release_per_tick": _release_initial(parameters),
            "line_target_slew_per_tick": parameters.line_target_slew_per_tick,
            "line_turn_sensitivity": parameters.line_turn_sensitivity,
            "maximum_racing_line_offset_ratio": parameters.maximum_racing_line_offset_ratio,
            "normal_steer_limit": parameters.normal_steer_limit,
            "preview_line_compensation": parameters.preview_line_compensation,
            "racing_line_entry_offset_ratio": parameters.racing_line_entry_offset_ratio,
            "racing_line_exit_offset_ratio": parameters.racing_line_exit_offset_ratio,
            "racing_line_offset_ratio": parameters.racing_line_offset_ratio,
            "side_slow_start_m": parameters.side_slow_start_m,
            "side_speed_floor": parameters.side_speed_floor,
            "startup_speed_cap_mps": parameters.startup_speed_cap_mps,
            "startup_speed_cap_seconds": parameters.startup_speed_cap_seconds,
            "steer_slew_per_tick": parameters.steer_slew_per_tick,
            "steering_speed_reduction": parameters.steering_speed_reduction,
            "straight_target_speed_mps": parameters.straight_target_speed_mps,
            "throttle_gain": parameters.throttle_gain,
            "wall_balance_gain": parameters.wall_balance_gain,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
            "yaw_damping_gain": parameters.yaw_damping_gain,
            "yaw_speed_reduction": parameters.yaw_speed_reduction,
        }
    if preset in ("faster-line-v10", "faster-line-v11", "faster-line-v12", "faster-line-v13"):
        return {
            "center_steer_gain": parameters.center_steer_gain,
            "front_brake_start_m": parameters.front_brake_start_m,
            "heading_steer_gain": parameters.heading_steer_gain,
            "line_clearance_m": parameters.line_clearance_m,
            "line_target_release_per_tick": _release_initial(parameters),
            "line_target_slew_per_tick": parameters.line_target_slew_per_tick,
            "line_turn_sensitivity": parameters.line_turn_sensitivity,
            "maximum_racing_line_offset_ratio": parameters.maximum_racing_line_offset_ratio,
            "normal_steer_limit": parameters.normal_steer_limit,
            "preview_line_compensation": parameters.preview_line_compensation,
            "racing_line_entry_offset_ratio": parameters.racing_line_entry_offset_ratio,
            "racing_line_exit_offset_ratio": parameters.racing_line_exit_offset_ratio,
            "racing_line_offset_ratio": parameters.racing_line_offset_ratio,
            "steer_slew_per_tick": parameters.steer_slew_per_tick,
            "straight_target_speed_mps": parameters.straight_target_speed_mps,
            "wall_balance_gain": parameters.wall_balance_gain,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
            "yaw_speed_reduction": parameters.yaw_speed_reduction,
        }
    if preset == "faster-line-v9":
        return {
            "center_steer_gain": parameters.center_steer_gain,
            "heading_steer_gain": parameters.heading_steer_gain,
            "line_clearance_m": parameters.line_clearance_m,
            "line_target_release_per_tick": _release_initial(parameters),
            "line_target_slew_per_tick": parameters.line_target_slew_per_tick,
            "line_turn_sensitivity": parameters.line_turn_sensitivity,
            "maximum_racing_line_offset_ratio": parameters.maximum_racing_line_offset_ratio,
            "normal_steer_limit": parameters.normal_steer_limit,
            "preview_line_compensation": parameters.preview_line_compensation,
            "racing_line_entry_offset_ratio": parameters.racing_line_entry_offset_ratio,
            "racing_line_exit_offset_ratio": parameters.racing_line_exit_offset_ratio,
            "racing_line_offset_ratio": parameters.racing_line_offset_ratio,
            "steer_slew_per_tick": parameters.steer_slew_per_tick,
            "steering_speed_reduction": parameters.steering_speed_reduction,
            "wall_balance_gain": parameters.wall_balance_gain,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
            "yaw_damping_gain": parameters.yaw_damping_gain,
            "yaw_speed_reduction": parameters.yaw_speed_reduction,
        }
    if preset == "faster-line-v8":
        return {
            "center_steer_gain": parameters.center_steer_gain,
            "heading_steer_gain": parameters.heading_steer_gain,
            "line_clearance_m": parameters.line_clearance_m,
            "line_target_release_per_tick": _release_initial(parameters),
            "line_target_slew_per_tick": parameters.line_target_slew_per_tick,
            "line_turn_sensitivity": parameters.line_turn_sensitivity,
            "maximum_racing_line_offset_ratio": parameters.maximum_racing_line_offset_ratio,
            "normal_steer_limit": parameters.normal_steer_limit,
            "preview_line_compensation": parameters.preview_line_compensation,
            "racing_line_entry_offset_ratio": parameters.racing_line_entry_offset_ratio,
            "racing_line_exit_offset_ratio": parameters.racing_line_exit_offset_ratio,
            "racing_line_offset_ratio": parameters.racing_line_offset_ratio,
            "steer_slew_per_tick": parameters.steer_slew_per_tick,
            "steering_speed_reduction": parameters.steering_speed_reduction,
            "wall_balance_gain": parameters.wall_balance_gain,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
            "yaw_damping_gain": parameters.yaw_damping_gain,
            "yaw_speed_reduction": parameters.yaw_speed_reduction,
        }
    if preset == "faster-line-v7":
        return {
            "center_steer_gain": parameters.center_steer_gain,
            "heading_steer_gain": parameters.heading_steer_gain,
            "line_clearance_m": parameters.line_clearance_m,
            "line_target_release_per_tick": _release_initial(parameters),
            "line_target_slew_per_tick": parameters.line_target_slew_per_tick,
            "line_turn_sensitivity": parameters.line_turn_sensitivity,
            "maximum_racing_line_offset_ratio": parameters.maximum_racing_line_offset_ratio,
            "normal_steer_limit": parameters.normal_steer_limit,
            "preview_line_compensation": parameters.preview_line_compensation,
            "racing_line_entry_offset_ratio": parameters.racing_line_entry_offset_ratio,
            "racing_line_exit_offset_ratio": parameters.racing_line_exit_offset_ratio,
            "racing_line_offset_ratio": parameters.racing_line_offset_ratio,
            "steer_slew_per_tick": parameters.steer_slew_per_tick,
            "wall_balance_gain": parameters.wall_balance_gain,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
            "yaw_damping_gain": parameters.yaw_damping_gain,
        }
    if preset == "faster-line-v6":
        return {
            "preview_line_compensation": parameters.preview_line_compensation,
            "wall_balance_line_compensation": parameters.wall_balance_line_compensation,
        }
    if preset not in ("faster-line-v2", "faster-line-v3", "faster-line-v4", "faster-line-v5"):
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
    evaluator_recycle_trials: int = DEFAULT_EVALUATOR_RECYCLE_TRIALS,
) -> tuple[CandidateEvaluation, ...]:
    tasks = tuple(
        EvaluationTask(
            candidate=candidate,
            parameter_names=parameter_names,
            base_parameters=base_parameters,
            seeds=seeds,
            duration_seconds=duration_seconds,
            evaluator_recycle_trials=evaluator_recycle_trials,
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
    if objective_kind == "lap-time":
        return lap_time_score(results, baseline_distances)
    if objective_kind == "lap-time-v2":
        return lap_time_score_v2(results, baseline_distances)
    if objective_kind == "lap-time-v3":
        return lap_time_score_v3(results, baseline_distances)
    if objective_kind == "lap-time-v4":
        return lap_time_score_v4(results, baseline_distances)
    if objective_kind == "lap-time-v5":
        return lap_time_score_v5(results, baseline_distances)
    if objective_kind == "lap-time-v6":
        return lap_time_score_v6(results, baseline_distances)
    if objective_kind == "lap-time-v8":
        return lap_time_score_v8(results, baseline_distances)
    if objective_kind == "lap-time-v9":
        return lap_time_score_v9(results, baseline_distances)
    if objective_kind == "lap-time-v7":
        return lap_time_score_v7(results, baseline_distances)
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
    global _worker_evaluator, _worker_trial_count
    if _worker_evaluator is None:
        _worker_evaluator = SoloEvaluator()
        _worker_trial_count = 0
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
    _worker_trial_count += len(task.seeds)
    if 0 < task.evaluator_recycle_trials <= _worker_trial_count:
        _worker_evaluator.close()
        _worker_evaluator = None
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
    if preset == "faster-line-v14":
        base = FASTER_LINE_V14_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v14_parameter_space(base)
    if preset == "faster-line-v15":
        base = FASTER_LINE_V15_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v15_parameter_space(base)
    if preset == "faster-line-v16":
        base = FASTER_LINE_V16_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v16_parameter_space(base)
    if preset == "faster-line-v17":
        base = FASTER_LINE_V17_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v17_parameter_space(base)
    if preset == "faster-line-v25":
        base = FASTER_LINE_V25_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v25_parameter_space(base)
    if preset == "faster-line-v24":
        base = FASTER_LINE_V24_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v24_parameter_space(base)
    if preset == "faster-line-v23":
        base = FASTER_LINE_V23_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v23_parameter_space(base)
    if preset == "faster-line-v22":
        base = FASTER_LINE_V22_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v22_parameter_space(base)
    if preset == "faster-line-v21":
        base = FASTER_LINE_V21_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v21_parameter_space(base)
    if preset == "faster-line-v20":
        base = FASTER_LINE_V20_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v20_parameter_space(base)
    if preset == "faster-line-v19":
        base = FASTER_LINE_V19_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v19_parameter_space(base)
    if preset == "faster-line-v18":
        base = FASTER_LINE_V18_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v18_parameter_space(base)
    if preset == "faster-line-v13":
        base = FASTER_LINE_V13_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v13_parameter_space(base)
    if preset == "faster-line-v12":
        base = FASTER_LINE_V12_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v12_parameter_space(base)
    if preset == "faster-line-v11":
        base = FASTER_LINE_V11_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v11_parameter_space(base)
    if preset == "faster-line-v10":
        base = FASTER_LINE_V10_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v10_parameter_space(base)
    if preset == "faster-line-v9":
        base = FASTER_LINE_V9_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v9_parameter_space(base)
    if preset == "faster-line-v8":
        base = FASTER_LINE_V8_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v8_parameter_space(base)
    if preset == "faster-line-v7":
        base = FASTER_LINE_V7_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v7_parameter_space(base)
    if preset == "faster-line-v6":
        base = FASTER_LINE_V6_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v6_parameter_space(base)
    if preset == "faster-line-v5":
        base = FASTER_LINE_V5_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v5_parameter_space(base)
    if preset == "faster-line-v4":
        base = FASTER_LINE_V4_BASE_PARAMETERS
        if seed_checkpoint is not None:
            base = replace(base, **_checkpoint_parameter_values(seed_checkpoint))
        return base, faster_line_v4_parameter_space(base)
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
    record = cast(dict[str, object], raw)
    vector = record.get("best_parameter_vector")
    if not isinstance(vector, dict):
        raise ValueError("seed checkpoint has no best parameter vector")
    context = record.get("checkpoint_context", {})
    if not isinstance(context, dict):
        raise ValueError("seed checkpoint context is invalid")
    fields = ControllerParameters.__dataclass_fields__
    values: dict[str, float] = {}
    # Fixed context is part of the candidate just as much as its searched
    # vector. Apply it first so a searched value wins if an older checkpoint
    # redundantly stores a name in both mappings.
    for source in (cast(dict[str, object], context), cast(dict[str, object], vector)):
        for name, value in source.items():
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
        choices=(
            "minimum",
            "faster",
            "faster-line",
            "faster-line-v2-probe",
            "faster-line-v2",
            "faster-line-v3",
            "faster-line-v4",
            "faster-line-v5",
            "faster-line-v6",
            "faster-line-v7",
            "faster-line-v8",
            "faster-line-v9",
            "faster-line-v10",
            "faster-line-v11",
            "faster-line-v12",
            "faster-line-v13",
            "faster-line-v14",
            "faster-line-v15",
            "faster-line-v16",
            "faster-line-v17",
            "faster-line-v18",
            "faster-line-v19",
            "faster-line-v20",
            "faster-line-v21",
            "faster-line-v22",
            "faster-line-v23",
            "faster-line-v24",
            "faster-line-v25",
        ),
    )
    parser.add_argument("--optimizer", choices=("cem", "ga"), default="cem")
    parser.add_argument(
        "--objective",
        choices=(
            "improved",
            "improved-v2",
            "lap-time",
            "lap-time-v2",
            "lap-time-v3",
            "lap-time-v4",
            "lap-time-v5",
            "lap-time-v6",
            "lap-time-v7",
            "lap-time-v8",
            "lap-time-v9",
        ),
        default="improved",
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--optimizer-seed", type=int, default=590_112)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument(
        "--evaluator-recycle-trials",
        type=int,
        default=DEFAULT_EVALUATOR_RECYCLE_TRIALS,
        help="rebuild each worker's evaluator after this many trials to cap memory growth; 0 disables",
    )
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
        evaluator_recycle_trials=cast(int, args.evaluator_recycle_trials),
        duration_seconds=float(args.seconds),
        worker_count=int(args.workers),
    )


if __name__ == "__main__":
    main()
