"""Dependency-free preview controller shared by the racing presets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, pi

from racing import CameraCompetitorReading, RobotCommand, RobotSensors

MAX_RACING_LINE_OFFSET_RATIO: float = 0.65
# A corner whose near/far local curvatures differ by at least this fraction is
# unambiguously in entry or exit.  The target slew limiter smooths the phase
# transition; blending entry directly with the opposite-sign apex target here
# made the requested outside line disappear before the car could reach it.
LINE_PHASE_TRANSITION_RATIO: float = 0.25
# Ignore straight-line camera noise when deciding whether one turn direction has
# persisted long enough to be treated as a sweeper.
SWEEPER_ACTIVE_CURVATURE: float = 0.02


class ControlMode(Enum):
    """High-level safety state for the preview controller."""

    NORMAL = "normal"
    AVOID = "avoid"
    RECOVER = "recover"


@dataclass(frozen=True, slots=True)
class ControllerParameters:
    """Immutable policy parameters suitable for hand tuning or search."""

    # Normalization only. Control decisions read PreviewFeatures.speed_mps, which is
    # never clamped, so raising a target speed above this cap stays safe.
    speed_cap_mps: float = 30.0
    yaw_rate_cap_degrees_per_s: float = 180.0
    center_offset_cap_m: float = 3.3
    heading_error_cap_degrees: float = 90.0
    lidar_cap_m: float = 20.0
    contact_cap_s: float = 1.0
    competitor_distance_cap_m: float = 20.0
    competitor_closing_speed_cap_mps: float = 12.0

    heading_steer_gain: float = 0.92
    center_steer_gain: float = 0.52
    yaw_damping_gain: float = 0.20
    straight_lookahead_weights: tuple[float, float, float] = (0.12, 0.30, 0.52)
    corner_lookahead_weights: tuple[float, float, float] = (0.52, 0.34, 0.12)
    racing_line_offset_ratio: float = 0.0
    racing_line_entry_offset_ratio: float = 0.0
    racing_line_exit_offset_ratio: float = 0.0
    phase_aware_racing_line: bool = False
    pose_invariant_racing_line: bool = False
    curvature_offset_compensation: float = 0.0
    curvature_heading_compensation: float = 0.0
    preview_line_compensation: float = 0.0
    wall_balance_line_compensation: float = 0.0
    # Clamp on the requested line offset, as a fraction of centre_offset_cap_m.
    # The car half-width is 0.63 m against a 3.3 m surface edge and a 4.7 m
    # barrier, so its body edge reaches the barrier at about 4.07 m of offset,
    # i.e. a ratio near 1.23.  Anything at or above that is geometric contact.
    maximum_racing_line_offset_ratio: float = MAX_RACING_LINE_OFFSET_RATIO
    # Feed the speed scalar the pose-invariant local curvature instead of the
    # distance-inflated `track_curvature_preview`.  Default-off: the promoted
    # vector's speed schedule is calibrated against the older signal.
    pose_invariant_speed_curvature: bool = False
    line_turn_sensitivity: float = 1.0
    line_target_slew_per_tick: float = 1.0
    # Seed the tracked racing-line target from the first available preview
    # instead of treating the centreline as prior state.  Steering still uses
    # its independent slew limiter.  Default-off preserves existing presets.
    initialize_line_target_from_preview: bool = False
    # Rate limit for the target moving back *toward* the centreline while staying
    # on the same side, separate from the outward rate above.  The phase target
    # is a function of instantaneous curvature, so it collapses to zero whenever
    # local curvature is low and drags the car back to the centre even when the
    # next corner bends the same way and it should simply stay out there.  On
    # this track the shortest in-corridor path sits pinned to one edge for 64 m
    # at a stretch, which the symmetric rate cannot express.  A crossing to the
    # opposite side still uses the fast outward rate, so response to a genuine
    # opposite corner is unchanged.  None means "same rate as the outward slew",
    # which reproduces the symmetric behaviour exactly.
    line_target_release_per_tick: float | None = None
    # Add speed only after one turn direction has persisted long enough to be a
    # broad sweeper.  The short hold carries the bonus through its exit without
    # raising the global straight/corner targets.  All values are default-off.
    sweeper_minimum_duration_s: float = 0.0
    sweeper_speed_hold_seconds: float = 0.0
    sweeper_target_speed_bonus_mps: float = 0.0
    # Preview a broad sweeper from the entry geometry instead of waiting until
    # the turn has already persisted.  The far local curvature must dominate
    # the opposite-sign near curvature and sit inside this band.  All four
    # values are default-off so existing presets remain bit-for-bit unchanged.
    sweeper_preview_minimum_far_curvature: float = 0.0
    sweeper_preview_maximum_far_curvature: float = 0.0
    sweeper_preview_speed_hold_seconds: float = 0.0
    sweeper_preview_target_speed_bonus_mps: float = 0.0
    # Add speed in proportion to the pose-invariant corner-exit phase.  The
    # bonus is default-off and shares the existing non-stacking bonus envelope.
    corner_exit_target_speed_bonus_mps: float = 0.0
    # Add pace only after the pose-invariant preview has remained nearly straight
    # for long enough to identify a real corridor.  This raises the target on the
    # long final stretch without raising every approach speed.  All values are
    # default-off so existing presets are unchanged.
    long_straight_minimum_duration_s: float = 0.0
    long_straight_maximum_local_curvature: float = 0.0
    long_straight_speed_bonus_seconds: float = 0.0
    long_straight_target_speed_bonus_mps: float = 0.0
    line_clearance_m: float = 0.0
    wall_balance_gain: float = 0.38
    normal_steer_limit: float = 0.92
    steer_slew_per_tick: float = 0.075
    emergency_steer_slew_per_tick: float = 0.24

    curvature_heading_degrees: float = 52.0
    curvature_lateral_ratio: float = 0.58
    straight_target_speed_mps: float = 11.0
    corner_target_speed_mps: float = 6.4
    # Optional launch-only cap.  The first corner can arrive before the tracked
    # racing-line target has settled from its initial zero state; limiting only
    # that approach avoids paying for a wall-recovery correction without
    # slowing later laps.  Either zero value disables the behavior.
    startup_speed_cap_mps: float = 0.0
    startup_speed_cap_seconds: float = 0.0
    # Optional one-shot rotation pulse for the opening corner.  This is keyed to
    # local wall/steering geometry rather than a seed or world pose: while the car
    # is fast, committed to a turn, and closing on the front wall, briefly request
    # reverse throttle to brake all four wheels while holding the turn.  The normal
    # drive-latch release below inserts the required neutral tick afterward.  A
    # short zero-steer phase can then arrest the rotation before power comes back.
    # Zero brake or zero window/pulse duration disables the behavior exactly.
    startup_drift_brake: float = 0.0
    startup_drift_window_seconds: float = 0.0
    startup_drift_trigger_front_m: float = 8.5
    startup_drift_minimum_speed_mps: float = 14.0
    startup_drift_minimum_steer: float = 0.30
    startup_drift_pulse_seconds: float = 0.0
    startup_drift_steer_gain: float = 1.0
    startup_drift_straighten_seconds: float = 0.0
    steering_speed_reduction: float = 0.28
    yaw_speed_reduction: float = 0.22
    # Distance at which the throttle starts lifting, not braking; the name predates
    # coasting and is kept because `minimum_parameter_space` checkpoints record it.
    front_brake_start_m: float = 8.0
    front_stop_m: float = 1.4
    side_slow_start_m: float = 1.35
    side_speed_floor: float = 0.58
    throttle_gain: float = 0.20
    # Reachable from AVOID and RECOVER only; NORMAL lifts instead of braking.
    brake_gain: float = 0.30
    maximum_forward_throttle: float = 0.82
    maximum_brake: float = 0.90
    drive_latch_release_speed_mps: float = 0.30

    avoid_front_wall_m: float = 3.2
    avoid_diagonal_wall_m: float = 1.55
    avoid_side_wall_m: float = 0.95
    avoid_speed_mps: float = 3.4
    avoid_steer_gain: float = 1.05
    competitor_avoid_distance_m: float = 4.5
    competitor_slow_distance_m: float = 8.0
    competitor_half_angle_degrees: float = 42.0
    competitor_speed_floor: float = 0.45
    competitor_racing_steer_fraction: float = 0.20
    competitor_pass_steer_gain: float = 0.75

    stall_speed_mps: float = 0.28
    stall_timeout_s: float = 0.9
    recovery_blocked_front_m: float = 0.72
    recovery_reverse_throttle: float = -0.72
    recovery_steer: float = 0.78
    recovery_minimum_s: float = 0.65
    recovery_maximum_s: float = 1.35
    recovery_exit_heading_degrees: float = 75.0
    recovery_exit_front_m: float = 1.25


@dataclass(frozen=True, slots=True)
class PreviewFeatures:
    """Finite, normalized controller inputs derived from public sensors."""

    speed: float
    speed_mps: float
    yaw_rate: float
    center_offset: float
    heading_error: float
    lookahead_offsets: tuple[float, float, float]
    lookahead_distances_m: tuple[float, float, float]
    wall_lidar: tuple[float, ...]
    lidar: tuple[float, ...]
    wall_front: float
    wall_front_left: float
    wall_front_right: float
    wall_left: float
    wall_right: float
    lidar_front: float
    lidar_front_left: float
    lidar_front_right: float
    contact: float
    has_competitor_ahead: bool
    competitor_distance: float
    competitor_angle: float
    competitor_closing_speed: float


@dataclass(slots=True)
class ControllerState:
    """Mutable history owned by one controller instance."""

    mode: ControlMode = ControlMode.NORMAL
    previous_steer: float = 0.0
    previous_throttle: float = 0.0
    previous_distance_m: float | None = None
    stalled_seconds: float = 0.0
    recovery_seconds: float = 0.0
    recovery_steer: float = 0.0
    line_target: float = 0.0
    line_target_initialized: bool = False
    sweeper_turn_direction: float = 0.0
    sweeper_turn_seconds: float = 0.0
    sweeper_speed_hold_direction: float = 0.0
    sweeper_speed_hold_seconds: float = 0.0
    sweeper_preview_hold_direction: float = 0.0
    sweeper_preview_speed_hold_seconds: float = 0.0
    corner_exit_speed_bonus_factor: float = 0.0
    long_straight_seconds: float = 0.0
    long_straight_drift_armed: bool = False
    long_straight_drift_direction: float = 0.0
    long_straight_drift_seconds_remaining: float = 0.0
    startup_drift_attempted: bool = False
    startup_drift_direction: float = 0.0
    startup_drift_seconds_remaining: float = 0.0
    startup_drift_straighten_seconds_remaining: float = 0.0


@dataclass(frozen=True, slots=True)
class PreviewDiagnostics:
    """Opt-in controller internals consumed by the training trace tool."""

    features: PreviewFeatures
    mode: ControlMode
    kappa: tuple[float, float, float]
    line_shape: tuple[float, float]
    line_target: float
    curvature: float
    target_speed_mps: float
    heading_steer: float
    center_line_steer: float
    preview_steer: float
    yaw_damping_steer: float
    wall_balance_steer: float
    desired_steer: float
    command: RobotCommand


@dataclass(slots=True)
class PreviewController:
    """Preview-line controller with bounded steering and safety recovery."""

    parameters: ControllerParameters
    state: ControllerState = field(default_factory=ControllerState)
    diagnostics_sink: Callable[[PreviewDiagnostics], None] | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        features = build_preview_features(sensors, self.parameters)
        self._update_stall_state(sensors, features)
        self._select_mode(features, sensors.dt_s)

        startup_speed_cap_mps: float | None = None
        startup_elapsed_s = max(0, sensors.tick) * _clamp(_finite_or(sensors.dt_s), 0.0, 0.25)
        if (
            self.parameters.startup_speed_cap_mps > 0.0
            and self.parameters.startup_speed_cap_seconds > 0.0
            and startup_elapsed_s < self.parameters.startup_speed_cap_seconds
        ):
            startup_speed_cap_mps = self.parameters.startup_speed_cap_mps

        racing_steer, curvature, steering_diagnostics = self._preview_steer(features, sensors.dt_s)
        if self.state.mode is ControlMode.RECOVER:
            desired_steer = self.state.recovery_steer
            throttle = self.parameters.recovery_reverse_throttle
            target_speed_mps = 0.0
        elif self.state.mode is ControlMode.AVOID:
            desired_steer = self._avoidance_steer(features, racing_steer)
            throttle, target_speed_mps = self._speed_command(
                features,
                curvature,
                desired_steer,
                avoiding=True,
                startup_speed_cap_mps=startup_speed_cap_mps,
            )
        else:
            desired_steer = racing_steer
            throttle, target_speed_mps = self._speed_command(
                features,
                curvature,
                desired_steer,
                avoiding=False,
                startup_speed_cap_mps=startup_speed_cap_mps,
            )
            desired_steer, throttle = self._startup_drift_command(
                features,
                racing_steer,
                throttle,
                startup_elapsed_s=startup_elapsed_s,
                dt_s=sensors.dt_s,
            )
            desired_steer, throttle = self._post_long_straight_drift_command(
                features,
                desired_steer,
                throttle,
                dt_s=sensors.dt_s,
            )

        steer_slew = (
            self.parameters.steer_slew_per_tick
            if self.state.mode is ControlMode.NORMAL
            else self.parameters.emergency_steer_slew_per_tick
        )
        steer = _slew(self.state.previous_steer, desired_steer, steer_slew)
        throttle = self._release_drive_latch(throttle, features.speed_mps)
        command = RobotCommand(
            throttle=_clamp(_finite_or(throttle), -1.0, 1.0),
            steer=_clamp(_finite_or(steer), -1.0, 1.0),
        )
        self.state.previous_throttle = command.throttle
        self.state.previous_steer = command.steer
        if self.diagnostics_sink is not None and steering_diagnostics is not None:
            (
                kappa,
                line_shape,
                line_target,
                heading_steer,
                center_line_steer,
                preview_steer,
                yaw_damping_steer,
                wall_balance_steer,
            ) = steering_diagnostics
            self.diagnostics_sink(
                PreviewDiagnostics(
                    features=features,
                    mode=self.state.mode,
                    kappa=kappa,
                    line_shape=line_shape,
                    line_target=line_target,
                    curvature=curvature,
                    target_speed_mps=target_speed_mps,
                    heading_steer=heading_steer,
                    center_line_steer=center_line_steer,
                    preview_steer=preview_steer,
                    yaw_damping_steer=yaw_damping_steer,
                    wall_balance_steer=wall_balance_steer,
                    desired_steer=desired_steer,
                    command=command,
                )
            )
        return command

    def _startup_drift_command(
        self,
        features: PreviewFeatures,
        racing_steer: float,
        throttle: float,
        *,
        startup_elapsed_s: float,
        dt_s: float,
    ) -> tuple[float, float]:
        """Apply one short, geometry-triggered opening-corner rotation pulse."""
        parameters = self.parameters
        state = self.state
        enabled = (
            parameters.startup_drift_brake > 0.0
            and parameters.startup_drift_window_seconds > 0.0
            and parameters.startup_drift_pulse_seconds > 0.0
        )
        if not enabled:
            return racing_steer, throttle

        dt_s = _clamp(_finite_or(dt_s, 1.0 / 60.0), 0.0, 0.25)
        if not state.startup_drift_attempted:
            front_m = min(features.wall_front, features.lidar_front) * parameters.lidar_cap_m
            should_start = (
                startup_elapsed_s < parameters.startup_drift_window_seconds
                and features.speed_mps >= parameters.startup_drift_minimum_speed_mps
                and abs(racing_steer) >= parameters.startup_drift_minimum_steer
                and front_m <= parameters.startup_drift_trigger_front_m
            )
            if should_start:
                state.startup_drift_attempted = True
                state.startup_drift_direction = 1.0 if racing_steer >= 0.0 else -1.0
                state.startup_drift_seconds_remaining = parameters.startup_drift_pulse_seconds

        if state.startup_drift_seconds_remaining > 0.0:
            held_magnitude = max(abs(racing_steer), parameters.startup_drift_minimum_steer)
            desired_steer = state.startup_drift_direction * _clamp(
                held_magnitude * parameters.startup_drift_steer_gain,
                0.0,
                1.0,
            )
            state.startup_drift_seconds_remaining = max(0.0, state.startup_drift_seconds_remaining - dt_s)
            if state.startup_drift_seconds_remaining <= 0.0:
                state.startup_drift_straighten_seconds_remaining = parameters.startup_drift_straighten_seconds
            return desired_steer, -_clamp(parameters.startup_drift_brake, 0.0, 1.0)

        if state.startup_drift_straighten_seconds_remaining > 0.0:
            state.startup_drift_straighten_seconds_remaining = max(
                0.0,
                state.startup_drift_straighten_seconds_remaining - dt_s,
            )
            return 0.0, throttle
        return racing_steer, throttle

    def _release_drive_latch(self, throttle: float, speed_mps: float) -> float:
        """Clear the vehicle's pending-direction latch with one zero-throttle tick.

        After any negative throttle while rolling forward the simulator returns zero
        engine force and re-arms the latch every tick, even for positive requests,
        until the car reaches ~1 km/h. A throttle of exactly zero clears it. Reversing
        is excluded because brake-before-forward is correct and self-terminating there.
        """
        if (
            throttle > 0.0
            and self.state.previous_throttle < 0.0
            and speed_mps > self.parameters.drive_latch_release_speed_mps
        ):
            return 0.0
        return throttle

    def _post_long_straight_drift_command(
        self,
        features: PreviewFeatures,
        racing_steer: float,
        throttle: float,
        *,
        dt_s: float,
    ) -> tuple[float, float]:
        """Spend an armed corridor boost on a short rotation pulse next corner."""
        parameters = self.parameters
        state = self.state
        enabled = parameters.startup_drift_brake > 0.0 and parameters.startup_drift_pulse_seconds > 0.0
        if not enabled or state.startup_drift_seconds_remaining > 0.0:
            return racing_steer, throttle

        if state.long_straight_drift_seconds_remaining <= 0.0 and state.long_straight_drift_armed:
            front_m = min(features.wall_front, features.lidar_front) * parameters.lidar_cap_m
            should_start = (
                features.speed_mps >= parameters.startup_drift_minimum_speed_mps
                and abs(racing_steer) >= parameters.startup_drift_minimum_steer
                and front_m <= parameters.startup_drift_trigger_front_m
            )
            if should_start:
                state.long_straight_drift_armed = False
                state.long_straight_drift_direction = 1.0 if racing_steer >= 0.0 else -1.0
                state.long_straight_drift_seconds_remaining = parameters.startup_drift_pulse_seconds

        if state.long_straight_drift_seconds_remaining <= 0.0:
            return racing_steer, throttle
        held_magnitude = max(abs(racing_steer), parameters.startup_drift_minimum_steer)
        desired_steer = state.long_straight_drift_direction * _clamp(
            held_magnitude * parameters.startup_drift_steer_gain,
            0.0,
            1.0,
        )
        step_seconds = _clamp(_finite_or(dt_s, 1.0 / 60.0), 0.0, 0.25)
        state.long_straight_drift_seconds_remaining = max(
            0.0,
            state.long_straight_drift_seconds_remaining - step_seconds,
        )
        return desired_steer, -_clamp(parameters.startup_drift_brake, 0.0, 1.0)

    def _update_stall_state(self, sensors: RobotSensors, features: PreviewFeatures) -> None:
        distance_m = max(0.0, _finite_or(sensors.odometry.distance_m))
        previous_distance_m = self.state.previous_distance_m
        self.state.previous_distance_m = distance_m
        dt_s = _clamp(_finite_or(sensors.dt_s), 0.0, 0.25)
        moving = previous_distance_m is None or distance_m > previous_distance_m + 0.001
        # Coasting is a normal cruise state, so a zero throttle still counts as driving.
        trying_to_drive = self.state.previous_throttle >= 0.0
        if moving or abs(features.speed_mps) > self.parameters.stall_speed_mps or not trying_to_drive:
            self.state.stalled_seconds = 0.0
        else:
            self.state.stalled_seconds += dt_s

    def _select_mode(self, features: PreviewFeatures, dt_s: float) -> None:
        parameters = self.parameters
        contact_active = features.contact > 0.0
        blocked = features.wall_front * parameters.lidar_cap_m < parameters.recovery_blocked_front_m
        stalled = self.state.stalled_seconds >= parameters.stall_timeout_s

        if self.state.mode is ControlMode.RECOVER:
            self.state.recovery_seconds += _clamp(_finite_or(dt_s, 1.0 / 60.0), 0.0, 0.25)
            safe_to_exit = (
                not contact_active
                and features.wall_front * parameters.lidar_cap_m >= parameters.recovery_exit_front_m
                and abs(features.heading_error) * parameters.heading_error_cap_degrees
                <= parameters.recovery_exit_heading_degrees
            )
            if self.state.recovery_seconds < parameters.recovery_minimum_s:
                return
            if not safe_to_exit and self.state.recovery_seconds < parameters.recovery_maximum_s:
                return
            self.state.recovery_seconds = 0.0
            self.state.stalled_seconds = 0.0
        elif contact_active or blocked or stalled:
            self.state.mode = ControlMode.RECOVER
            self.state.recovery_seconds = 0.0
            self.state.recovery_steer = self._open_side_steer(features, parameters.recovery_steer)
            return

        self.state.mode = ControlMode.AVOID if self._avoidance_needed(features) else ControlMode.NORMAL

    def _preview_steer(
        self,
        features: PreviewFeatures,
        dt_s: float,
    ) -> tuple[
        float,
        float,
        tuple[tuple[float, float, float], tuple[float, float], float, float, float, float, float, float] | None,
    ]:
        parameters = self.parameters
        kappa = track_curvature_preview(features, parameters)
        line_shape = track_shape_preview(features, parameters)
        if parameters.pose_invariant_speed_curvature:
            lateral_curvature = max(abs(line_shape[0]), abs(line_shape[1]))
        else:
            lateral_curvature = max(abs(value) for value in kappa)
        heading_curvature = abs(features.heading_error) * (
            parameters.heading_error_cap_degrees / parameters.curvature_heading_degrees
        )
        curvature = _clamp(
            max(
                lateral_curvature / parameters.curvature_lateral_ratio,
                heading_curvature,
            ),
            0.0,
            1.0,
        )
        weights = tuple(
            straight + (corner - straight) * curvature
            for straight, corner in zip(
                parameters.straight_lookahead_weights,
                parameters.corner_lookahead_weights,
                strict=True,
            )
        )
        curve_direction = _clamp(
            features.heading_error * 0.25
            + features.lookahead_offsets[0] * 0.20
            + features.lookahead_offsets[1] * 0.25
            + features.lookahead_offsets[2] * 0.30,
            -1.0,
            1.0,
        )
        if parameters.pose_invariant_racing_line:
            line_target = self._tracked_line_target(features, line_shape, dt_s)
        else:
            line_target = self._racing_line_offset(features, curvature, curve_direction)

        preview_shift_m = line_target * parameters.center_offset_cap_m * parameters.preview_line_compensation
        preview = sum(
            (offset + preview_shift_m / distance_m) * weight
            for offset, distance_m, weight in zip(
                features.lookahead_offsets,
                features.lookahead_distances_m,
                weights,
                strict=True,
            )
        )
        wall_shift = (
            line_target
            * 2.0
            * parameters.center_offset_cap_m
            / parameters.lidar_cap_m
            * parameters.wall_balance_line_compensation
            * (1.0 - curvature)
        )
        wall_balance = features.wall_right - features.wall_left + wall_shift
        heading_steer = features.heading_error * parameters.heading_steer_gain
        center_line_steer = (features.center_offset + line_target) * parameters.center_steer_gain
        yaw_damping_steer = -features.yaw_rate * parameters.yaw_damping_gain
        wall_balance_steer = wall_balance * parameters.wall_balance_gain
        desired = heading_steer + center_line_steer + preview + yaw_damping_steer + wall_balance_steer
        diagnostics = None
        if self.diagnostics_sink is not None:
            diagnostics = (
                kappa,
                line_shape,
                line_target,
                heading_steer,
                center_line_steer,
                preview,
                yaw_damping_steer,
                wall_balance_steer,
            )
        return (
            _clamp(desired, -parameters.normal_steer_limit, parameters.normal_steer_limit),
            curvature,
            diagnostics,
        )

    def _tracked_line_target(
        self,
        features: PreviewFeatures,
        shape: tuple[float, float],
        dt_s: float,
    ) -> float:
        parameters = self.parameters
        raw_target = self._line_target(shape)
        self._update_sweeper_speed_state(shape, dt_s)
        self._update_sweeper_preview_speed_state(shape, dt_s)
        self._update_corner_exit_speed_state(shape)
        self._update_long_straight_speed_state(shape, dt_s)
        previous_target = self.state.line_target
        if parameters.initialize_line_target_from_preview and not self.state.line_target_initialized:
            target = raw_target
        else:
            target = _slew(previous_target, raw_target, self._line_target_rate(previous_target, raw_target))
        self.state.line_target_initialized = True
        if parameters.line_clearance_m > 0.0 and target != 0.0:
            toward = features.wall_right if target > 0.0 else features.wall_left
            clearance_m = toward * parameters.lidar_cap_m
            if clearance_m < parameters.line_clearance_m:
                target *= _unit_interval(clearance_m / parameters.line_clearance_m)
        self.state.line_target = target
        return target

    def _update_sweeper_speed_state(
        self,
        shape: tuple[float, float],
        dt_s: float,
    ) -> None:
        """Arm the local speed bonus after a sustained broad turn."""
        parameters = self.parameters
        if (
            parameters.sweeper_target_speed_bonus_mps <= 0.0
            or parameters.sweeper_minimum_duration_s <= 0.0
            or parameters.sweeper_speed_hold_seconds <= 0.0
        ):
            return

        state = self.state
        step_seconds = _clamp(_finite_or(dt_s, 1.0 / 60.0), 0.0, 0.25)
        direction_signal = shape[0] + shape[1]
        direction = _sign(direction_signal) if abs(direction_signal) >= SWEEPER_ACTIVE_CURVATURE else 0.0

        if direction != 0.0:
            if direction == state.sweeper_turn_direction:
                state.sweeper_turn_seconds += step_seconds
            else:
                state.sweeper_turn_direction = direction
                state.sweeper_turn_seconds = step_seconds

            if direction != state.sweeper_speed_hold_direction and state.sweeper_speed_hold_seconds > 0.0:
                state.sweeper_speed_hold_seconds = 0.0
                state.sweeper_speed_hold_direction = 0.0

            if state.sweeper_turn_seconds >= parameters.sweeper_minimum_duration_s:
                state.sweeper_speed_hold_direction = direction
                state.sweeper_speed_hold_seconds = parameters.sweeper_speed_hold_seconds
        else:
            state.sweeper_turn_direction = 0.0
            state.sweeper_turn_seconds = 0.0

        state.sweeper_speed_hold_seconds = max(0.0, state.sweeper_speed_hold_seconds - step_seconds)

    def _update_sweeper_preview_speed_state(
        self,
        shape: tuple[float, float],
        dt_s: float,
    ) -> None:
        """Arm a speed bonus from the distinctive entry of a broad sweeper."""
        parameters = self.parameters
        if (
            parameters.sweeper_preview_minimum_far_curvature <= 0.0
            or parameters.sweeper_preview_maximum_far_curvature <= 0.0
            or parameters.sweeper_preview_speed_hold_seconds <= 0.0
            or parameters.sweeper_preview_target_speed_bonus_mps <= 0.0
        ):
            return

        state = self.state
        step_seconds = _clamp(_finite_or(dt_s, 1.0 / 60.0), 0.0, 0.25)
        near_turn, far_turn = shape
        direction_signal = near_turn + far_turn
        direction = _sign(direction_signal) if abs(direction_signal) >= SWEEPER_ACTIVE_CURVATURE else 0.0
        if (
            direction != 0.0
            and state.sweeper_preview_hold_direction != 0.0
            and direction != state.sweeper_preview_hold_direction
        ):
            state.sweeper_preview_speed_hold_seconds = 0.0
            state.sweeper_preview_hold_direction = 0.0

        far_magnitude = abs(far_turn)
        preview_entry = (
            near_turn * far_turn < 0.0
            and far_magnitude > abs(near_turn)
            and parameters.sweeper_preview_minimum_far_curvature
            <= far_magnitude
            <= parameters.sweeper_preview_maximum_far_curvature
        )
        if preview_entry:
            state.sweeper_preview_hold_direction = _sign(far_turn)
            state.sweeper_preview_speed_hold_seconds = parameters.sweeper_preview_speed_hold_seconds
        else:
            state.sweeper_preview_speed_hold_seconds = max(
                0.0,
                state.sweeper_preview_speed_hold_seconds - step_seconds,
            )
            if state.sweeper_preview_speed_hold_seconds == 0.0:
                state.sweeper_preview_hold_direction = 0.0

    def _update_corner_exit_speed_state(self, shape: tuple[float, float]) -> None:
        """Track how strongly the local preview indicates a corner exit."""
        if self.parameters.corner_exit_target_speed_bonus_mps <= 0.0:
            self.state.corner_exit_speed_bonus_factor = 0.0
            return
        near_turn, far_turn = shape
        dominant = max(abs(near_turn), abs(far_turn))
        if dominant <= 1e-6:
            self.state.corner_exit_speed_bonus_factor = 0.0
            return
        phase_difference = (abs(far_turn) - abs(near_turn)) / dominant
        self.state.corner_exit_speed_bonus_factor = _unit_interval(-phase_difference / LINE_PHASE_TRANSITION_RATIO)

    def _update_long_straight_speed_state(
        self,
        shape: tuple[float, float],
        dt_s: float,
    ) -> None:
        """Measure a sustained locally straight corridor without world coordinates."""
        parameters = self.parameters
        if (
            parameters.long_straight_minimum_duration_s <= 0.0
            or parameters.long_straight_maximum_local_curvature <= 0.0
            or parameters.long_straight_speed_bonus_seconds <= 0.0
            or parameters.long_straight_target_speed_bonus_mps <= 0.0
        ):
            self.state.long_straight_seconds = 0.0
            return
        if max(abs(shape[0]), abs(shape[1])) <= parameters.long_straight_maximum_local_curvature:
            step_seconds = _clamp(_finite_or(dt_s, 1.0 / 60.0), 0.0, 0.25)
            self.state.long_straight_seconds += step_seconds
        else:
            self.state.long_straight_seconds = 0.0

    def _line_target_rate(self, previous_target: float, raw_target: float) -> float:
        """Pick the outward or the release rate for this tick's target move."""
        parameters = self.parameters
        release = parameters.line_target_release_per_tick
        if release is None:
            return parameters.line_target_slew_per_tick
        relaxing_on_the_same_side = abs(raw_target) < abs(previous_target) and raw_target * previous_target >= 0.0
        return release if relaxing_on_the_same_side else parameters.line_target_slew_per_tick

    def _line_target(self, shape: tuple[float, float]) -> float:
        parameters = self.parameters
        near_turn, far_turn = shape
        near_magnitude = abs(near_turn)
        far_magnitude = abs(far_turn)
        dominant = max(near_magnitude, far_magnitude)
        if dominant <= 1e-6:
            return 0.0

        phase_difference = (far_magnitude - near_magnitude) / dominant
        entry_weight = _unit_interval(phase_difference / LINE_PHASE_TRANSITION_RATIO)
        exit_weight = _unit_interval(-phase_difference / LINE_PHASE_TRANSITION_RATIO)
        apex_weight = max(0.0, 1.0 - entry_weight - exit_weight)
        turn_strength = _unit_interval(dominant / max(1e-6, parameters.line_turn_sensitivity))
        target = turn_strength * (
            -_sign(far_turn) * entry_weight * parameters.racing_line_entry_offset_ratio
            + _sign(near_turn) * apex_weight * parameters.racing_line_offset_ratio
            - _sign(near_turn) * exit_weight * parameters.racing_line_exit_offset_ratio
        )
        limit = parameters.maximum_racing_line_offset_ratio
        return _clamp(target, -limit, limit)

    def _racing_line_offset(
        self,
        features: PreviewFeatures,
        curvature: float,
        curve_direction: float,
    ) -> float:
        parameters = self.parameters
        if not parameters.phase_aware_racing_line:
            return curve_direction * curvature * parameters.racing_line_offset_ratio

        near_turn = _clamp(
            features.heading_error * 0.55 + features.lookahead_offsets[0] * 0.30 + features.lookahead_offsets[1] * 0.15,
            -1.0,
            1.0,
        )
        far_turn = _clamp(
            features.lookahead_offsets[0] * 0.15
            + features.lookahead_offsets[1] * 0.30
            + features.lookahead_offsets[2] * 0.55,
            -1.0,
            1.0,
        )
        near_magnitude = abs(near_turn)
        far_magnitude = abs(far_turn)
        dominant_magnitude = max(near_magnitude, far_magnitude)
        if dominant_magnitude <= 1e-6:
            return 0.0

        entry_weight = max(0.0, (far_magnitude - near_magnitude) / dominant_magnitude)
        exit_weight = max(0.0, (near_magnitude - far_magnitude) / dominant_magnitude)
        apex_weight = max(0.0, 1.0 - entry_weight - exit_weight)
        offset = curvature * (
            -_sign(far_turn) * entry_weight * parameters.racing_line_entry_offset_ratio
            + _sign(curve_direction) * apex_weight * parameters.racing_line_offset_ratio
            - _sign(near_turn) * exit_weight * parameters.racing_line_exit_offset_ratio
        )
        return _clamp(offset, -MAX_RACING_LINE_OFFSET_RATIO, MAX_RACING_LINE_OFFSET_RATIO)

    def _avoidance_needed(self, features: PreviewFeatures) -> bool:
        parameters = self.parameters
        lidar_cap_m = parameters.lidar_cap_m
        return (
            features.wall_front * lidar_cap_m < parameters.avoid_front_wall_m
            or min(features.wall_front_left, features.wall_front_right) * lidar_cap_m < parameters.avoid_diagonal_wall_m
            or min(features.wall_left, features.wall_right) * lidar_cap_m < parameters.avoid_side_wall_m
            or (
                features.has_competitor_ahead
                and features.competitor_distance * parameters.competitor_distance_cap_m
                < parameters.competitor_avoid_distance_m
            )
        )

    def _avoidance_steer(self, features: PreviewFeatures, racing_steer: float) -> float:
        parameters = self.parameters
        left_opening = 0.65 * features.wall_front_left + 0.35 * features.wall_left
        right_opening = 0.65 * features.wall_front_right + 0.35 * features.wall_right
        opening_balance = right_opening - left_opening
        wall_hazard = (
            features.wall_front * parameters.lidar_cap_m < parameters.avoid_front_wall_m
            or min(features.wall_front_left, features.wall_front_right) * parameters.lidar_cap_m
            < parameters.avoid_diagonal_wall_m
            or min(features.wall_left, features.wall_right) * parameters.lidar_cap_m < parameters.avoid_side_wall_m
        )
        wall_steer = 0.0
        if wall_hazard:
            if abs(opening_balance) < 0.03:
                opening_balance = 1.0 if racing_steer >= 0.0 else -1.0
            wall_steer = _clamp(opening_balance * parameters.avoid_steer_gain, -1.0, 1.0)

        competitor_steer = 0.0
        if features.has_competitor_ahead:
            if abs(features.competitor_angle) > 0.05:
                competitor_steer = 1.0 if features.competitor_angle < 0.0 else -1.0
            else:
                competitor_steer = 1.0 if right_opening >= left_opening else -1.0
        desired = (
            racing_steer * parameters.competitor_racing_steer_fraction
            + wall_steer
            + competitor_steer * parameters.competitor_pass_steer_gain
        )
        return _clamp(desired, -1.0, 1.0)

    def _speed_command(
        self,
        features: PreviewFeatures,
        curvature: float,
        steer: float,
        *,
        avoiding: bool,
        startup_speed_cap_mps: float | None,
    ) -> tuple[float, float]:
        parameters = self.parameters
        target_speed = (
            parameters.straight_target_speed_mps
            + (parameters.corner_target_speed_mps - parameters.straight_target_speed_mps) * curvature
        )
        speed_bonus = 0.0
        if self.state.sweeper_speed_hold_seconds > 0.0:
            speed_bonus = max(speed_bonus, parameters.sweeper_target_speed_bonus_mps)
        if self.state.sweeper_preview_speed_hold_seconds > 0.0:
            speed_bonus = max(speed_bonus, parameters.sweeper_preview_target_speed_bonus_mps)
        if (
            parameters.long_straight_target_speed_bonus_mps > 0.0
            and parameters.long_straight_minimum_duration_s > 0.0
            and parameters.long_straight_minimum_duration_s
            <= self.state.long_straight_seconds
            < parameters.long_straight_minimum_duration_s + parameters.long_straight_speed_bonus_seconds
        ):
            speed_bonus = max(speed_bonus, parameters.long_straight_target_speed_bonus_mps)
            if startup_speed_cap_mps is None:
                self.state.long_straight_drift_armed = True
        speed_bonus = max(
            speed_bonus,
            parameters.corner_exit_target_speed_bonus_mps * self.state.corner_exit_speed_bonus_factor,
        )
        target_speed += max(0.0, speed_bonus)
        target_speed *= 1.0 - parameters.steering_speed_reduction * abs(steer)
        target_speed *= 1.0 - parameters.yaw_speed_reduction * abs(features.yaw_rate)

        front_m = min(features.wall_front, features.lidar_front) * parameters.lidar_cap_m
        if front_m < parameters.front_brake_start_m:
            front_factor = _unit_interval(
                (front_m - parameters.front_stop_m) / (parameters.front_brake_start_m - parameters.front_stop_m)
            )
            target_speed *= front_factor

        side_m = min(features.wall_left, features.wall_right) * parameters.lidar_cap_m
        if side_m < parameters.side_slow_start_m:
            side_factor = parameters.side_speed_floor + (1.0 - parameters.side_speed_floor) * _unit_interval(
                side_m / parameters.side_slow_start_m
            )
            target_speed *= side_factor

        if features.has_competitor_ahead:
            distance_m = features.competitor_distance * parameters.competitor_distance_cap_m
            competitor_factor = parameters.competitor_speed_floor + (
                1.0 - parameters.competitor_speed_floor
            ) * _unit_interval(distance_m / parameters.competitor_slow_distance_m)
            closing_factor = 1.0 - 0.25 * max(0.0, features.competitor_closing_speed)
            target_speed *= competitor_factor * closing_factor

        if avoiding:
            target_speed = min(target_speed, parameters.avoid_speed_mps)
        if startup_speed_cap_mps is not None:
            target_speed = min(target_speed, startup_speed_cap_mps)

        speed_error = target_speed - features.speed_mps
        if speed_error >= 0.0:
            return (
                _clamp(
                    speed_error * parameters.throttle_gain,
                    0.0,
                    parameters.maximum_forward_throttle,
                ),
                target_speed,
            )
        if not avoiding:
            # Lifting is the only sane decelerator here. A negative throttle costs the
            # whole 100 N engine until the car reaches ~1 km/h and buys ~1.9 N of brake.
            return 0.0, target_speed
        return -_clamp(-speed_error * parameters.brake_gain, 0.0, parameters.maximum_brake), target_speed

    def _open_side_steer(self, features: PreviewFeatures, magnitude: float) -> float:
        left_opening = features.wall_left + features.wall_front_left
        right_opening = features.wall_right + features.wall_front_right
        return -abs(magnitude) if left_opening > right_opening else abs(magnitude)


def build_preview_features(sensors: RobotSensors, parameters: ControllerParameters) -> PreviewFeatures:
    """Normalize every controller input and replace non-finite values."""
    camera = sensors.camera
    lookahead_offsets, lookahead_distances_m = _normalized_lookahead_offsets(sensors, parameters)
    wall_lidar = tuple(_normalized_distance(value, parameters.lidar_cap_m) for value in sensors.wall_lidar.distances_m)
    lidar = tuple(_normalized_distance(value, parameters.lidar_cap_m) for value in sensors.lidar.distances_m)
    competitor = _nearest_ahead_competitor(sensors, parameters)
    if competitor is None:
        competitor_distance = 1.0
        competitor_angle = 0.0
        competitor_closing_speed = 0.0
    else:
        competitor_distance = _normalized_distance(competitor.distance_m, parameters.competitor_distance_cap_m)
        competitor_angle = _clamp(
            _finite_or(competitor.angle_degrees) / parameters.competitor_half_angle_degrees,
            -1.0,
            1.0,
        )
        competitor_closing_speed = _clamp(
            _finite_or(competitor.closing_speed_mps) / parameters.competitor_closing_speed_cap_mps,
            -1.0,
            1.0,
        )

    speed_mps = _finite_or(sensors.odometry.speed_mps)
    return PreviewFeatures(
        speed=_clamp(speed_mps / parameters.speed_cap_mps, -1.0, 1.0),
        speed_mps=speed_mps,
        yaw_rate=_clamp(
            _finite_or(sensors.imu.yaw_rate_degrees_per_s) / parameters.yaw_rate_cap_degrees_per_s,
            -1.0,
            1.0,
        ),
        center_offset=(
            _clamp(
                _finite_or(camera.center_offset_m) / parameters.center_offset_cap_m,
                -1.0,
                1.0,
            )
            if camera.visible
            else 0.0
        ),
        heading_error=(
            _clamp(
                _finite_or(camera.heading_error_degrees) / parameters.heading_error_cap_degrees,
                -1.0,
                1.0,
            )
            if camera.visible
            else 0.0
        ),
        lookahead_offsets=lookahead_offsets,
        lookahead_distances_m=lookahead_distances_m,
        wall_lidar=wall_lidar,
        lidar=lidar,
        wall_front=_normalized_distance(sensors.wall_lidar.front_m, parameters.lidar_cap_m),
        wall_front_left=_normalized_distance(sensors.wall_lidar.front_left_m, parameters.lidar_cap_m),
        wall_front_right=_normalized_distance(sensors.wall_lidar.front_right_m, parameters.lidar_cap_m),
        wall_left=_normalized_distance(sensors.wall_lidar.left_m, parameters.lidar_cap_m),
        wall_right=_normalized_distance(sensors.wall_lidar.right_m, parameters.lidar_cap_m),
        lidar_front=_normalized_distance(sensors.lidar.front_m, parameters.lidar_cap_m),
        lidar_front_left=_normalized_distance(sensors.lidar.front_left_m, parameters.lidar_cap_m),
        lidar_front_right=_normalized_distance(sensors.lidar.front_right_m, parameters.lidar_cap_m),
        contact=_clamp(_finite_or(sensors.contact.any_contact) / parameters.contact_cap_s, 0.0, 1.0),
        has_competitor_ahead=competitor is not None,
        competitor_distance=competitor_distance,
        competitor_angle=competitor_angle,
        competitor_closing_speed=competitor_closing_speed,
    )


def _normalized_lookahead_offsets(
    sensors: RobotSensors,
    parameters: ControllerParameters,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    defaults = (4.0, 9.0, 16.0)
    if not sensors.camera.visible:
        return (0.0, 0.0, 0.0), defaults
    normalized: list[float] = []
    resolved_distances_m: list[float] = []
    for index in range(3):
        offset_m = sensors.camera.lookahead_offsets_m[index] if index < len(sensors.camera.lookahead_offsets_m) else 0.0
        lookahead_m = (
            sensors.camera.lookahead_distances_m[index]
            if index < len(sensors.camera.lookahead_distances_m)
            else defaults[index]
        )
        scale_m = max(0.1, abs(_finite_or(lookahead_m, defaults[index])))
        normalized.append(_clamp(_finite_or(offset_m) / scale_m, -1.0, 1.0))
        resolved_distances_m.append(scale_m)
    return (
        (normalized[0], normalized[1], normalized[2]),
        (resolved_distances_m[0], resolved_distances_m[1], resolved_distances_m[2]),
    )


def track_curvature_preview(
    features: PreviewFeatures,
    parameters: ControllerParameters,
) -> tuple[float, float, float]:
    """Return upcoming centreline shape independent of the car's own pose."""
    heading_radians = (
        features.heading_error
        * parameters.heading_error_cap_degrees
        * (pi / 180.0)
        * parameters.curvature_heading_compensation
    )
    offset_m = features.center_offset * parameters.center_offset_cap_m * parameters.curvature_offset_compensation
    values = tuple(
        _clamp(offset_norm - offset_m / distance_m - heading_radians, -1.0, 1.0)
        for offset_norm, distance_m in zip(
            features.lookahead_offsets,
            features.lookahead_distances_m,
            strict=True,
        )
    )
    return values[0], values[1], values[2]


def track_shape_preview(
    features: PreviewFeatures,
    parameters: ControllerParameters,
) -> tuple[float, float]:
    """Return local centreline curvature at a near and a far preview depth.

    The camera reports where the centreline point at each lookahead distance
    sits relative to the car, so ``c_i`` accumulates every degree of turn over
    ``[0, d_i]``.  Dividing by ``d_i`` therefore does *not* yield curvature: for
    a constant-radius corner ``c_i ~ d_i^2 / 2R``, so ``c_i / d_i ~ d_i / 2R``
    keeps growing with lookahead distance and the far preview outranks the near
    one everywhere inside a corner, whatever the phase.

    Differencing the segment slopes recovers a genuinely local curvature, and
    cancels the car's own pose while doing it.  A heading error tilts every
    slope by the same angle, so it drops out of both differences exactly.  A
    lateral displacement shifts every measured ``c_i`` by the same amount, so it
    drops out too once the chord is anchored at ``c(0)``, which the camera
    already reports as the offset to the nearest centreline point.  That makes
    this signal independent of ``curvature_offset_compensation`` and
    ``curvature_heading_compensation``, which exist only for the speed scalar.
    """
    near_m, mid_m, far_m = features.lookahead_distances_m
    if near_m <= 0.0 or mid_m <= near_m or far_m <= mid_m:
        return 0.0, 0.0
    anchor_m = features.center_offset * parameters.center_offset_cap_m
    near_offset_m = features.lookahead_offsets[0] * near_m
    mid_offset_m = features.lookahead_offsets[1] * mid_m
    far_offset_m = features.lookahead_offsets[2] * far_m
    near_slope = (near_offset_m - anchor_m) / near_m
    mid_slope = (mid_offset_m - near_offset_m) / (mid_m - near_m)
    far_slope = (far_offset_m - mid_offset_m) / (far_m - mid_m)
    return (
        2.0 * (mid_slope - near_slope) / mid_m,
        2.0 * (far_slope - mid_slope) / (far_m - near_m),
    )


def _nearest_ahead_competitor(
    sensors: RobotSensors,
    parameters: ControllerParameters,
) -> CameraCompetitorReading | None:
    nearest: CameraCompetitorReading | None = None
    nearest_distance = float("inf")
    for competitor in sensors.camera.competitors:
        angle_degrees = _finite_or(competitor.angle_degrees, 180.0)
        distance_m = max(0.0, _finite_or(competitor.distance_m, parameters.competitor_distance_cap_m))
        if abs(angle_degrees) <= parameters.competitor_half_angle_degrees and distance_m < nearest_distance:
            nearest = competitor
            nearest_distance = distance_m
    return nearest


def _normalized_distance(value: float, cap_m: float) -> float:
    return _clamp(_finite_or(value, cap_m) / cap_m, 0.0, 1.0)


def _finite_or(value: float, fallback: float = 0.0) -> float:
    return value if isfinite(value) else fallback


def _unit_interval(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _slew(previous: float, desired: float, maximum_change: float) -> float:
    return _clamp(desired, previous - maximum_change, previous + maximum_change)
