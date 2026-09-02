from __future__ import annotations

from dataclasses import replace
from math import cos, isfinite, radians, sin

import pytest

from controllers.minimum_viable import create_controller
from controllers.preview_controller import (
    LINE_PHASE_TRANSITION_RATIO,
    MAX_RACING_LINE_OFFSET_RATIO,
    ControllerParameters,
    ControlMode,
    PreviewController,
    PreviewFeatures,
    TransitionDriftPhase,
    build_preview_features,
    track_curvature_preview,
    track_shape_preview,
)
from racing import (
    CameraCompetitorReading,
    CameraSensors,
    ContactSensors,
    ImuSensors,
    LidarSensors,
    OdometrySensors,
    RobotSensors,
)
from racing.race.progress import TrackProgressModel, track_pose_at_distance, track_progress_model_for_layout
from racing.track.world import TRACK_ID_MUGELLO_SHORT, TrackPoint

# The camera's fixed preview depths; the shape signal differences across them.
TRACK_LOOKAHEAD_DISTANCES_M = (4.0, 9.0, 16.0)


def lidar(*distances_m: float) -> LidarSensors:
    return LidarSensors(distances_m=distances_m)


def test_preview_features_replace_infinity_and_clamp_inputs() -> None:
    sensors = RobotSensors(
        imu=ImuSensors(yaw_rate_degrees_per_s=-1_000.0),
        odometry=OdometrySensors(speed_mps=100.0),
        wall_lidar=lidar(*(float("inf") for _ in range(7))),
        lidar=lidar(-2.0, 5.0, float("inf"), 500.0, 2.0, 1.0, 0.0),
        camera=CameraSensors(
            center_offset_m=-100.0,
            heading_error_degrees=500.0,
            lookahead_offsets_m=(-100.0, 0.0, 100.0),
        ),
        contact=ContactSensors(any_contact=10.0),
    )

    features = build_preview_features(sensors, ControllerParameters())

    assert features.speed == 1.0
    assert features.speed_mps == 100.0
    assert features.yaw_rate == -1.0
    assert features.center_offset == -1.0
    assert features.heading_error == 1.0
    assert features.lookahead_offsets == (-1.0, 0.0, 1.0)
    assert features.wall_lidar == (1.0,) * 7
    assert features.lidar == (0.0, 0.25, 1.0, 1.0, 0.1, 0.05, 0.0)
    assert features.contact == 1.0


def test_preview_controller_steers_toward_track_preview_with_slew_limit() -> None:
    controller = PreviewController(ControllerParameters())
    right_curve = RobotSensors(
        camera=CameraSensors(
            center_offset_m=1.0,
            heading_error_degrees=25.0,
            lookahead_offsets_m=(1.5, 3.0, 5.0),
        )
    )

    first = controller(right_curve)
    second = controller(right_curve)

    assert first.steer == controller.parameters.steer_slew_per_tick
    assert second.steer > first.steer
    assert second.steer - first.steer <= controller.parameters.steer_slew_per_tick


def test_racing_line_offset_moves_toward_inside_of_curve() -> None:
    center_controller = PreviewController(ControllerParameters(steer_slew_per_tick=1.0))
    racing_controller = PreviewController(ControllerParameters(racing_line_offset_ratio=0.2, steer_slew_per_tick=1.0))
    right_curve = RobotSensors(
        camera=CameraSensors(
            heading_error_degrees=15.0,
            lookahead_offsets_m=(0.8, 2.5, 5.0),
        )
    )

    center_command = center_controller(right_curve)
    racing_command = racing_controller(right_curve)

    assert racing_command.steer > center_command.steer


def test_disabled_phase_aware_line_preserves_existing_racing_line() -> None:
    existing = PreviewController(
        ControllerParameters(
            racing_line_offset_ratio=0.2,
            steer_slew_per_tick=1.0,
        )
    )
    disabled = PreviewController(
        ControllerParameters(
            racing_line_offset_ratio=0.2,
            racing_line_entry_offset_ratio=0.65,
            racing_line_exit_offset_ratio=0.65,
            phase_aware_racing_line=False,
            steer_slew_per_tick=1.0,
        )
    )
    sensors = RobotSensors(
        camera=CameraSensors(
            heading_error_degrees=15.0,
            lookahead_offsets_m=(0.8, 2.5, 5.0),
        )
    )

    assert disabled(sensors) == existing(sensors)


def test_phase_aware_line_moves_outside_on_corner_entry_for_both_directions() -> None:
    for direction in (-1.0, 1.0):
        controller = _isolated_line_controller(entry=0.65)
        sensors = RobotSensors(
            camera=CameraSensors(
                lookahead_offsets_m=tuple(direction * value for value in (0.2, 1.8, 4.8)),
            )
        )

        command = controller(sensors)

        assert command.steer * direction < 0.0


def test_phase_aware_line_moves_inside_at_apex_for_both_directions() -> None:
    for direction in (-1.0, 1.0):
        controller = _isolated_line_controller(apex=0.65)
        sensors = RobotSensors(
            camera=CameraSensors(
                heading_error_degrees=direction * 18.0,
                lookahead_offsets_m=tuple(direction * value for value in (1.0, 2.7, 4.8)),
            )
        )

        command = controller(sensors)

        assert command.steer * direction > 0.0


def test_phase_aware_line_moves_outside_on_corner_exit_for_both_directions() -> None:
    for direction in (-1.0, 1.0):
        controller = _isolated_line_controller(exit=0.65)
        sensors = RobotSensors(
            camera=CameraSensors(
                heading_error_degrees=direction * 27.0,
                lookahead_offsets_m=tuple(direction * value for value in (1.2, 0.9, 0.8)),
            )
        )

        command = controller(sensors)

        assert command.steer * direction < 0.0


def test_phase_aware_line_is_neutral_on_a_straight_and_clamps_offset() -> None:
    straight_controller = _isolated_line_controller(apex=100.0, entry=100.0, exit=100.0)
    apex_controller = _isolated_line_controller(apex=100.0)

    straight_command = straight_controller(RobotSensors())
    apex_command = apex_controller(
        RobotSensors(
            camera=CameraSensors(
                heading_error_degrees=27.0,
                lookahead_offsets_m=(1.2, 2.7, 4.8),
            )
        )
    )

    assert straight_command.steer == 0.0
    assert apex_command.steer == 0.65


def test_preview_controller_lifts_instead_of_braking_above_target_speed() -> None:
    controller = PreviewController(ControllerParameters())

    command = controller(RobotSensors(odometry=OdometrySensors(speed_mps=16.0)))

    assert command.throttle == 0.0


def test_normal_mode_never_commands_negative_throttle_at_any_speed() -> None:
    controller = PreviewController(ControllerParameters())

    for speed_tenths in range(0, 401, 5):
        command = controller(RobotSensors(odometry=OdometrySensors(speed_mps=speed_tenths / 10.0)))
        assert controller.state.mode is ControlMode.NORMAL
        assert command.throttle >= 0.0


def test_speed_command_uses_unclamped_speed_above_the_normalization_cap() -> None:
    controller = PreviewController(ControllerParameters(speed_cap_mps=18.0, straight_target_speed_mps=20.0))

    command = controller(RobotSensors(odometry=OdometrySensors(speed_mps=25.0)))

    assert command.throttle == 0.0


def test_startup_speed_cap_only_limits_the_configured_opening_seconds() -> None:
    parameters = ControllerParameters(
        straight_target_speed_mps=25.0,
        startup_speed_cap_mps=18.0,
        startup_speed_cap_seconds=3.5,
        throttle_gain=1.0,
    )
    early = PreviewController(parameters)
    late = PreviewController(parameters)

    early_command = early(
        RobotSensors(
            dt_s=1.0 / 60.0,
            tick=120,
            odometry=OdometrySensors(speed_mps=19.0),
        )
    )
    late_command = late(
        RobotSensors(
            dt_s=1.0 / 60.0,
            tick=240,
            odometry=OdometrySensors(speed_mps=19.0),
        )
    )

    assert early_command.throttle == 0.0
    assert late_command.throttle > 0.0


def test_startup_drift_brakes_once_then_releases_latch_and_straightens() -> None:
    controller = PreviewController(
        ControllerParameters(
            straight_target_speed_mps=25.0,
            corner_target_speed_mps=25.0,
            steering_speed_reduction=0.0,
            throttle_gain=1.0,
            steer_slew_per_tick=1.0,
            startup_drift_brake=0.45,
            startup_drift_window_seconds=3.0,
            startup_drift_trigger_front_m=10.0,
            startup_drift_minimum_speed_mps=14.0,
            startup_drift_minimum_steer=0.20,
            startup_drift_pulse_seconds=1.0 / 60.0,
            startup_drift_steer_gain=1.5,
            startup_drift_straighten_seconds=0.10,
        )
    )
    sensors = RobotSensors(
        dt_s=1.0 / 60.0,
        tick=90,
        odometry=OdometrySensors(speed_mps=15.0),
        wall_lidar=lidar(*(9.0 for _ in range(7))),
        lidar=lidar(*(9.0 for _ in range(7))),
        camera=CameraSensors(
            heading_error_degrees=25.0,
            lookahead_offsets_m=(1.5, 3.0, 5.0),
        ),
    )

    drift = controller(sensors)
    release = controller(replace(sensors, tick=91))
    drive = controller(replace(sensors, tick=92))

    assert drift.throttle == -0.45
    assert drift.steer > 0.20
    assert controller.state.startup_drift_attempted
    # A positive request immediately after reverse throttle must become one exact
    # neutral tick so the simulator's pending-direction latch clears.
    assert release.throttle == 0.0
    assert drive.throttle > 0.0
    assert release.steer == 0.0
    assert drive.steer == 0.0


def test_startup_drift_default_off_preserves_normal_nonnegative_throttle() -> None:
    controller = PreviewController(ControllerParameters(straight_target_speed_mps=25.0))
    sensors = RobotSensors(
        dt_s=1.0 / 60.0,
        tick=90,
        odometry=OdometrySensors(speed_mps=15.0),
        wall_lidar=lidar(*(8.0 for _ in range(7))),
        lidar=lidar(*(8.0 for _ in range(7))),
        camera=CameraSensors(heading_error_degrees=25.0, lookahead_offsets_m=(1.5, 3.0, 5.0)),
    )

    assert controller(sensors).throttle >= 0.0
    assert not controller.state.startup_drift_attempted


def test_long_straight_bonus_arms_after_sustained_local_straight_and_resets() -> None:
    parameters = ControllerParameters(
        straight_target_speed_mps=20.0,
        corner_target_speed_mps=20.0,
        steering_speed_reduction=0.0,
        yaw_speed_reduction=0.0,
        throttle_gain=1.0,
        long_straight_minimum_duration_s=0.05,
        long_straight_maximum_local_curvature=0.01,
        long_straight_speed_bonus_seconds=0.10,
        long_straight_target_speed_bonus_mps=4.0,
    )
    controller = PreviewController(parameters)
    features = build_preview_features(RobotSensors(odometry=OdometrySensors(speed_mps=21.0)), parameters)

    _, before_target = controller._speed_command(  # pyright: ignore[reportPrivateUsage]
        features,
        0.0,
        0.0,
        avoiding=False,
        startup_speed_cap_mps=None,
    )
    for _ in range(3):
        controller._update_long_straight_speed_state(  # pyright: ignore[reportPrivateUsage]
            (0.002, -0.003), 1.0 / 60.0
        )
    throttle, boosted_target = controller._speed_command(  # pyright: ignore[reportPrivateUsage]
        features,
        0.0,
        0.0,
        avoiding=False,
        startup_speed_cap_mps=None,
    )

    assert before_target == 20.0
    assert boosted_target == 24.0
    assert throttle > 0.0
    assert controller.state.long_straight_drift_armed
    controller._update_long_straight_speed_state(  # pyright: ignore[reportPrivateUsage]
        (0.02, 0.0), 1.0 / 60.0
    )
    assert controller.state.long_straight_seconds == 0.0


def test_corridor_boost_arms_drift_for_the_next_hard_corner() -> None:
    parameters = ControllerParameters(
        startup_drift_brake=0.45,
        startup_drift_trigger_front_m=9.0,
        startup_drift_minimum_speed_mps=14.0,
        startup_drift_minimum_steer=0.20,
        startup_drift_pulse_seconds=0.05,
        startup_drift_steer_gain=1.5,
    )
    controller = PreviewController(parameters)
    controller.state.long_straight_drift_armed = True
    features = build_preview_features(
        RobotSensors(
            odometry=OdometrySensors(speed_mps=15.0),
            wall_lidar=lidar(*(8.0 for _ in range(7))),
            lidar=lidar(*(8.0 for _ in range(7))),
        ),
        parameters,
    )

    steer, throttle = controller._post_long_straight_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        -0.50,
        1.0,
        startup_elapsed_s=4.0,
        dt_s=1.0 / 60.0,
    )

    assert throttle == -0.45
    assert steer == -0.75
    assert not controller.state.long_straight_drift_armed
    assert controller.state.long_straight_drift_seconds_remaining > 0.0


def test_corridor_drift_can_override_the_opening_pulse_without_changing_it() -> None:
    parameters = ControllerParameters(
        startup_drift_brake=0.45,
        startup_drift_trigger_front_m=9.0,
        startup_drift_minimum_speed_mps=14.0,
        startup_drift_minimum_steer=0.20,
        startup_drift_pulse_seconds=0.05,
        startup_drift_steer_gain=1.5,
        long_straight_drift_brake=0.80,
        long_straight_drift_trigger_front_m=14.0,
        long_straight_drift_minimum_steer=0.08,
        long_straight_drift_pulse_seconds=0.15,
        long_straight_drift_override_after_seconds=10.0 / 3.0,
    )
    controller = PreviewController(parameters)
    controller.state.long_straight_drift_armed = True
    features = build_preview_features(
        RobotSensors(
            odometry=OdometrySensors(speed_mps=15.0),
            wall_lidar=lidar(*(12.0 for _ in range(7))),
            lidar=lidar(*(12.0 for _ in range(7))),
        ),
        parameters,
    )

    opening_steer, opening_throttle = controller._post_long_straight_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        -0.10,
        1.0,
        startup_elapsed_s=3.0,
        dt_s=1.0 / 60.0,
    )
    steer, throttle = controller._post_long_straight_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        -0.10,
        1.0,
        startup_elapsed_s=4.0,
        dt_s=1.0 / 60.0,
    )

    assert opening_throttle == 1.0
    assert opening_steer == -0.10
    assert throttle == -0.80
    assert steer == pytest.approx(-0.15)
    assert controller.state.long_straight_drift_seconds_remaining > parameters.startup_drift_pulse_seconds
    assert parameters.startup_drift_brake == 0.45
    assert parameters.startup_drift_trigger_front_m == 9.0


def test_transition_drift_carries_speed_rotates_aligns_and_countersteers() -> None:
    parameters = ControllerParameters(
        straight_target_speed_mps=20.0,
        corner_target_speed_mps=20.0,
        steering_speed_reduction=0.0,
        yaw_speed_reduction=0.0,
        throttle_gain=1.0,
        transition_drift_minimum_speed_mps=22.0,
        transition_drift_preview_curvature=0.04,
        transition_drift_trigger_curvature=0.01,
        transition_drift_preview_seconds=0.75,
        transition_drift_target_speed_bonus_mps=6.0,
        transition_drift_brake=0.50,
        transition_drift_steer=1.0,
        transition_drift_pulse_seconds=1.0 / 60.0,
        transition_drift_coast_max_seconds=0.20,
        transition_drift_alignment_heading_degrees=3.0,
        transition_drift_countersteer=0.70,
        transition_drift_countersteer_max_seconds=0.12,
        transition_drift_alignment_yaw_rate_degrees_per_s=60.0,
        transition_drift_cooldown_seconds=2.0,
    )
    controller = PreviewController(parameters)
    moving = RobotSensors(
        dt_s=1.0 / 60.0,
        odometry=OdometrySensors(speed_mps=23.0),
        imu=ImuSensors(yaw_rate_degrees_per_s=-100.0),
        camera=CameraSensors(heading_error_degrees=10.0),
    )
    features = build_preview_features(moving, parameters)

    steer, throttle = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        (-0.01, 0.06),
        0.20,
        0.0,
        dt_s=moving.dt_s,
    )
    assert (steer, throttle) == (0.20, 0.0)
    assert controller.state.transition_drift_phase is TransitionDriftPhase.APPROACH

    throttle, target_speed = controller._speed_command(  # pyright: ignore[reportPrivateUsage]
        features,
        curvature=0.0,
        steer=0.0,
        avoiding=False,
        startup_speed_cap_mps=None,
    )
    assert target_speed == 20.0
    assert throttle == 0.0

    controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        (0.08, -0.06),
        0.20,
        1.0,
        dt_s=moving.dt_s,
    )
    assert controller.state.transition_drift_phase is TransitionDriftPhase.ARMED

    rotate_steer, rotate_throttle = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        (-0.02, -0.10),
        0.20,
        1.0,
        dt_s=moving.dt_s,
    )
    assert (rotate_steer, rotate_throttle) == (-1.0, -0.50)
    assert controller.state.transition_drift_phase is TransitionDriftPhase.COAST

    _, target_speed = controller._speed_command(  # pyright: ignore[reportPrivateUsage]
        features,
        curvature=0.0,
        steer=0.0,
        avoiding=False,
        startup_speed_cap_mps=None,
    )
    assert target_speed == 26.0

    coast_steer, coast_throttle = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=moving.dt_s,
    )
    assert (coast_steer, coast_throttle) == (0.0, 1.0)

    countersteer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features,
        (0.02, 0.06),
        -0.20,
        1.0,
        dt_s=moving.dt_s,
    )
    assert countersteer == 0.70
    assert controller.state.transition_drift_phase is TransitionDriftPhase.COUNTERSTEER

    yaw_aligned = build_preview_features(
        replace(
            moving,
            imu=ImuSensors(yaw_rate_degrees_per_s=-30.0),
            camera=CameraSensors(heading_error_degrees=1.0),
        ),
        parameters,
    )
    released_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        yaw_aligned,
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=moving.dt_s,
    )
    assert released_steer == -0.20
    assert controller.state.transition_drift_phase is TransitionDriftPhase.IDLE
    assert controller.state.transition_drift_cooldown_seconds == 2.0


def test_transition_drift_releases_without_countersteer_when_aligned() -> None:
    parameters = ControllerParameters(
        transition_drift_minimum_speed_mps=22.0,
        transition_drift_preview_curvature=0.04,
        transition_drift_trigger_curvature=0.01,
        transition_drift_preview_seconds=0.75,
        transition_drift_brake=0.50,
        transition_drift_pulse_seconds=1.0 / 60.0,
        transition_drift_coast_max_seconds=0.20,
        transition_drift_alignment_heading_degrees=3.0,
        transition_drift_countersteer=0.70,
        transition_drift_countersteer_max_seconds=0.12,
        transition_drift_cooldown_seconds=2.0,
    )
    controller = PreviewController(parameters)
    controller.state.transition_drift_phase = TransitionDriftPhase.COAST
    controller.state.transition_drift_direction = -1.0
    controller.state.transition_drift_seconds_remaining = 0.20
    aligned = build_preview_features(
        RobotSensors(
            dt_s=1.0 / 60.0,
            odometry=OdometrySensors(speed_mps=23.0),
            camera=CameraSensors(heading_error_degrees=2.0),
        ),
        parameters,
    )

    released_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        aligned,
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=1.0 / 60.0,
    )
    assert released_steer == -0.20
    assert controller.state.transition_drift_phase is TransitionDriftPhase.IDLE
    assert controller.state.transition_drift_cooldown_seconds == 2.0


def test_transition_drift_can_hold_same_direction_settle_and_accelerate() -> None:
    parameters = ControllerParameters(
        straight_target_speed_mps=20.0,
        corner_target_speed_mps=20.0,
        steering_speed_reduction=0.0,
        yaw_speed_reduction=0.0,
        throttle_gain=1.0,
        transition_drift_minimum_speed_mps=22.0,
        transition_drift_preview_curvature=0.04,
        transition_drift_trigger_curvature=0.01,
        transition_drift_preview_seconds=0.75,
        transition_drift_target_speed_bonus_mps=5.0,
        transition_drift_brake=0.25,
        transition_drift_pulse_seconds=1.0 / 60.0,
        transition_drift_coast_max_seconds=0.10,
        transition_drift_same_direction_hold_trigger_yaw_rate_degrees_per_s=120.0,
        transition_drift_same_direction_hold=0.35,
        transition_drift_same_direction_hold_seconds=0.05,
        transition_drift_settle_max_seconds=0.10,
        transition_drift_alignment_yaw_rate_degrees_per_s=60.0,
        transition_drift_edge_acceleration_seconds=0.20,
        transition_drift_cooldown_seconds=2.0,
    )
    controller = PreviewController(parameters)
    controller.state.transition_drift_phase = TransitionDriftPhase.COAST
    controller.state.transition_drift_direction = -1.0
    controller.state.transition_drift_seconds_remaining = 0.10

    def features(yaw_rate: float) -> PreviewFeatures:
        return build_preview_features(
            RobotSensors(
                dt_s=0.05,
                odometry=OdometrySensors(speed_mps=23.0),
                imu=ImuSensors(yaw_rate_degrees_per_s=yaw_rate),
            ),
            parameters,
        )

    coast_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features(-100.0),
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=0.05,
    )
    assert coast_steer == 0.0
    assert controller.state.transition_drift_phase is TransitionDriftPhase.COAST

    hold_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features(-100.0),
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=0.05,
    )
    assert hold_steer == -0.35
    assert controller.state.transition_drift_phase is TransitionDriftPhase.HOLD

    held_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features(-100.0),
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=0.05,
    )
    assert held_steer == -0.35
    assert controller.state.transition_drift_phase is TransitionDriftPhase.SETTLE

    settle_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features(-100.0),
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=0.05,
    )
    assert settle_steer == 0.0
    assert controller.state.transition_drift_phase is TransitionDriftPhase.SETTLE

    edge_steer, _ = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        features(-30.0),
        (-0.05, -0.08),
        -0.20,
        1.0,
        dt_s=0.05,
    )
    assert edge_steer == -0.20
    assert controller.state.transition_drift_phase is TransitionDriftPhase.EDGE_ACCEL

    _, target_speed = controller._speed_command(  # pyright: ignore[reportPrivateUsage]
        features(-30.0),
        curvature=0.0,
        steer=0.0,
        avoiding=False,
        startup_speed_cap_mps=None,
    )
    assert target_speed == 25.0

    for _ in range(5):
        controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
            features(-30.0),
            (-0.05, -0.08),
            -0.20,
            1.0,
            dt_s=0.05,
        )
    assert controller.state.transition_drift_phase is TransitionDriftPhase.IDLE
    assert controller.state.transition_drift_cooldown_seconds == 2.0


def test_transition_drift_requires_the_favorable_entry_heading() -> None:
    parameters = ControllerParameters(
        transition_drift_minimum_speed_mps=22.0,
        transition_drift_preview_curvature=0.04,
        transition_drift_trigger_curvature=0.01,
        transition_drift_preview_seconds=0.75,
        transition_drift_brake=0.50,
        transition_drift_pulse_seconds=1.0 / 60.0,
        transition_drift_minimum_heading_error_degrees=12.0,
        transition_drift_cooldown_seconds=2.0,
    )
    controller = PreviewController(parameters)
    controller.state.transition_drift_phase = TransitionDriftPhase.ARMED
    controller.state.transition_drift_direction = -1.0
    controller.state.transition_drift_seconds_remaining = 0.50
    shallow_entry = build_preview_features(
        RobotSensors(
            dt_s=1.0 / 60.0,
            odometry=OdometrySensors(speed_mps=23.0),
            camera=CameraSensors(heading_error_degrees=5.0),
        ),
        parameters,
    )

    steer, throttle = controller._transition_drift_command(  # pyright: ignore[reportPrivateUsage]
        shallow_entry,
        (-0.02, -0.08),
        0.25,
        1.0,
        dt_s=1.0 / 60.0,
    )

    assert (steer, throttle) == (0.25, 1.0)
    assert controller.state.transition_drift_phase is TransitionDriftPhase.IDLE
    assert controller.state.transition_drift_cooldown_seconds == 2.0


def test_controller_still_brakes_during_avoidance() -> None:
    controller = PreviewController(ControllerParameters())
    sensors = RobotSensors(
        wall_lidar=lidar(0.4, 0.7, 1.0, 2.0, 8.0, 12.0, 15.0),
        lidar=lidar(0.4, 0.7, 1.0, 2.0, 8.0, 12.0, 15.0),
        odometry=OdometrySensors(speed_mps=4.0),
    )

    command = controller(sensors)

    assert controller.state.mode is ControlMode.AVOID
    assert command.throttle < 0.0


def test_controller_still_reverses_during_recovery() -> None:
    controller = PreviewController(ControllerParameters())
    sensors = RobotSensors(
        contact=ContactSensors(wall=0.1),
        wall_lidar=lidar(8.0, 8.0, 8.0, 5.0, 3.0, 2.0, 1.0),
    )

    command = controller(sensors)

    assert controller.state.mode is ControlMode.RECOVER
    assert command.throttle == controller.parameters.recovery_reverse_throttle


def test_positive_throttle_after_braking_inserts_one_zero_release_tick() -> None:
    controller = PreviewController(ControllerParameters())
    blocked = RobotSensors(
        wall_lidar=lidar(0.4, 0.7, 1.0, 2.0, 8.0, 12.0, 15.0),
        lidar=lidar(0.4, 0.7, 1.0, 2.0, 8.0, 12.0, 15.0),
        odometry=OdometrySensors(speed_mps=10.0),
    )
    clear = RobotSensors(odometry=OdometrySensors(speed_mps=1.0, distance_m=1.0))

    assert controller(blocked).throttle < 0.0
    assert controller(clear).throttle == 0.0
    assert controller(clear).throttle > 0.0


def test_release_tick_is_skipped_while_reversing() -> None:
    controller = PreviewController(ControllerParameters())
    controller.state.previous_throttle = -0.5

    command = controller(RobotSensors(odometry=OdometrySensors(speed_mps=-2.0)))

    assert command.throttle > 0.0


def test_stall_detection_uses_metres_per_second_not_normalized_speed() -> None:
    controller = PreviewController(ControllerParameters(stall_timeout_s=0.3))
    rolling = RobotSensors(dt_s=0.1, odometry=OdometrySensors(speed_mps=3.0, distance_m=0.0))

    for _ in range(5):
        controller(rolling)

    assert controller.state.stalled_seconds == 0.0
    assert controller.state.mode is not ControlMode.RECOVER


def test_preview_controller_avoids_a_blocked_side_toward_open_space() -> None:
    controller = PreviewController(ControllerParameters())
    sensors = RobotSensors(
        wall_lidar=lidar(0.4, 0.7, 1.0, 2.0, 8.0, 12.0, 15.0),
        lidar=lidar(0.4, 0.7, 1.0, 2.0, 8.0, 12.0, 15.0),
        odometry=OdometrySensors(speed_mps=4.0),
    )

    command = controller(sensors)

    assert controller.state.mode is ControlMode.AVOID
    assert command.steer > 0.0
    assert command.throttle < 0.0


def test_preview_controller_recovers_from_contact() -> None:
    controller = PreviewController(ControllerParameters())
    sensors = RobotSensors(
        contact=ContactSensors(wall=0.1),
        wall_lidar=lidar(8.0, 8.0, 8.0, 5.0, 3.0, 2.0, 1.0),
    )

    command = controller(sensors)

    assert controller.state.mode is ControlMode.RECOVER
    assert command.throttle < 0.0
    assert command.steer < 0.0


def test_preview_controller_recovers_after_sustained_stall() -> None:
    controller = PreviewController(ControllerParameters(stall_timeout_s=0.3))
    sensors = RobotSensors(dt_s=0.1, odometry=OdometrySensors(speed_mps=0.0, distance_m=0.0))

    command = controller(sensors)
    for _ in range(4):
        command = controller(sensors)

    assert controller.state.mode is ControlMode.RECOVER
    assert command.throttle < 0.0


def test_competitor_ahead_reduces_speed_and_selects_passing_side() -> None:
    clear_controller = PreviewController(ControllerParameters())
    traffic_controller = PreviewController(ControllerParameters())
    clear = RobotSensors(odometry=OdometrySensors(speed_mps=3.0))
    traffic = RobotSensors(
        odometry=OdometrySensors(speed_mps=3.0),
        camera=CameraSensors(
            competitors=(
                CameraCompetitorReading(distance_m=3.0, angle_degrees=-8.0, speed_mps=1.0, closing_speed_mps=2.0),
            )
        ),
    )

    clear_command = clear_controller(clear)
    traffic_command = traffic_controller(traffic)

    assert traffic_controller.state.mode is ControlMode.AVOID
    assert traffic_command.steer > 0.0
    assert traffic_command.throttle < clear_command.throttle


def test_competitor_on_right_selects_left_passing_side() -> None:
    controller = PreviewController(ControllerParameters())
    sensors = RobotSensors(
        camera=CameraSensors(
            competitors=(CameraCompetitorReading(distance_m=3.0, angle_degrees=8.0),),
        ),
    )

    command = controller(sensors)

    assert controller.state.mode is ControlMode.AVOID
    assert command.steer < 0.0


def test_factory_instances_do_not_share_steering_state() -> None:
    first = create_controller()
    second = create_controller()
    sensors = RobotSensors(camera=CameraSensors(heading_error_degrees=45.0, lookahead_offsets_m=(2.0, 4.0, 7.0)))

    first_command = first(sensors)
    first(sensors)
    second_command = second(sensors)

    assert first is not second
    assert first.state is not second.state
    assert first_command == second_command


def test_extreme_inputs_always_return_finite_normalized_commands() -> None:
    controller = PreviewController(ControllerParameters())
    sensors = RobotSensors(
        imu=ImuSensors(yaw_rate_degrees_per_s=float("inf")),
        odometry=OdometrySensors(speed_mps=float("-inf"), distance_m=float("inf")),
        camera=CameraSensors(
            center_offset_m=float("inf"),
            heading_error_degrees=float("-inf"),
            lookahead_offsets_m=(float("inf"), float("-inf"), 1e300),
        ),
    )

    for _ in range(120):
        command = controller(sensors)
        assert isfinite(command.throttle)
        assert isfinite(command.steer)
        assert -1.0 <= command.throttle <= 1.0
        assert -1.0 <= command.steer <= 1.0


def test_pose_invariant_curvature_ignores_lateral_offset_and_heading() -> None:
    heading_degrees = 10.0
    heading_radians = heading_degrees * 3.141592653589793 / 180.0
    distances = (4.0, 9.0, 16.0)
    sensors = RobotSensors(
        camera=CameraSensors(
            center_offset_m=2.0,
            heading_error_degrees=heading_degrees,
            lookahead_offsets_m=tuple(2.0 + heading_radians * distance for distance in distances),
            lookahead_distances_m=distances,
        )
    )
    parameters = ControllerParameters(curvature_offset_compensation=1.0, curvature_heading_compensation=1.0)

    features = build_preview_features(sensors, parameters)

    assert all(abs(value) < 1e-12 for value in track_curvature_preview(features, parameters))


def test_car_pre_positions_outside_on_a_straight_when_a_corner_is_ahead() -> None:
    controller = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.02)
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.30, 2.20)))

    command = controller(sensors)
    for _ in range(29):
        command = controller(sensors)

    assert controller.state.line_target < -0.30
    assert command.steer < 0.0


def test_whole_steering_law_commits_to_the_entry_target_before_turn_in() -> None:
    parameters = ControllerParameters(
        racing_line_offset_ratio=0.18784801522317213,
        racing_line_entry_offset_ratio=0.35,
        racing_line_exit_offset_ratio=0.14280170405453152,
        phase_aware_racing_line=True,
        pose_invariant_racing_line=True,
        curvature_offset_compensation=1.0,
        curvature_heading_compensation=1.0,
        preview_line_compensation=1.0,
        wall_balance_line_compensation=1.0,
        line_turn_sensitivity=0.03,
        line_target_slew_per_tick=0.1,
        steer_slew_per_tick=1.0,
    )
    controller = PreviewController(parameters)
    approach = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.30, 2.20)))

    command = controller(approach)
    for _ in range(3):
        command = controller(approach)

    assert controller.state.line_target == -0.35
    assert command.steer < 0.0


def test_pre_positioned_car_is_not_dragged_back_to_the_centreline() -> None:
    target_m = 1.5
    controller = _v2_line_controller(entry=target_m / ControllerParameters().center_offset_cap_m)
    sensors = RobotSensors(
        camera=CameraSensors(
            center_offset_m=target_m,
            lookahead_offsets_m=(target_m, target_m + 0.30, target_m + 2.20),
        )
    )

    command = controller(sensors)

    assert abs(command.steer) < 0.02


def test_target_offset_is_a_steering_equilibrium_of_the_whole_law() -> None:
    target_m = 2.0
    parameters = ControllerParameters(
        heading_steer_gain=0.0,
        center_steer_gain=1.0,
        yaw_damping_gain=0.0,
        preview_line_compensation=1.0,
        wall_balance_line_compensation=1.0,
        curvature_offset_compensation=1.0,
        wall_balance_gain=1.0,
        pose_invariant_racing_line=True,
        line_target_slew_per_tick=0.0,
        normal_steer_limit=1.0,
        steer_slew_per_tick=1.0,
    )
    controller = PreviewController(parameters)
    controller.state.line_target = target_m / parameters.center_offset_cap_m
    sensors = RobotSensors(
        camera=CameraSensors(
            center_offset_m=-target_m,
            lookahead_offsets_m=(-target_m, -target_m, -target_m),
        ),
        wall_lidar=lidar(12.0, 12.0, 12.0, 20.0, 8.0, 8.0, 8.0),
    )

    command = controller(sensors)

    assert abs(command.steer) < 0.01


def test_curvature_scalar_does_not_rise_from_the_car_s_own_lateral_offset() -> None:
    parameters = ControllerParameters(
        curvature_offset_compensation=1.0,
        pose_invariant_racing_line=True,
        steering_speed_reduction=0.0,
        yaw_speed_reduction=0.0,
    )
    centered = PreviewController(parameters)
    displaced = PreviewController(parameters)
    centered_sensors = RobotSensors(odometry=OdometrySensors(speed_mps=8.0))
    displaced_sensors = RobotSensors(
        odometry=OdometrySensors(speed_mps=8.0),
        camera=CameraSensors(center_offset_m=2.0, lookahead_offsets_m=(2.0, 2.0, 2.0)),
    )

    assert displaced(displaced_sensors).throttle == centered(centered_sensors).throttle


def test_high_center_steer_gain_tightens_line_tracking_not_centering() -> None:
    low = _v2_line_controller(entry=0.35, center_steer_gain=0.15)
    high = _v2_line_controller(entry=0.35, center_steer_gain=0.75)
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.30, 2.20)))

    low_command = low(sensors)
    high_command = high(sensors)

    assert high_command.steer < low_command.steer < 0.0


def test_line_target_is_rate_limited_across_ticks() -> None:
    controller = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.02)
    right_turn = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.30, 2.20)))
    left_turn = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, -0.30, -2.20)))
    controller(right_turn)
    before = controller.state.line_target

    controller(left_turn)

    assert controller.state.line_target - before <= 0.02


def test_line_target_retracts_when_target_side_wall_is_close_without_avoid() -> None:
    controller = _v2_line_controller(apex=0.65, line_clearance_m=1.8)
    sensors = RobotSensors(
        camera=CameraSensors(lookahead_offsets_m=(0.8, 3.6, 3.2)),
        wall_lidar=lidar(8.0, 8.0, 8.0, 20.0, 8.0, 8.0, 1.2),
    )

    controller(sensors)

    assert 0.0 < controller.state.line_target < 0.65
    assert controller.state.mode is ControlMode.NORMAL


def _isolated_line_controller(
    *,
    apex: float = 0.0,
    entry: float = 0.0,
    exit: float = 0.0,
) -> PreviewController:
    return PreviewController(
        ControllerParameters(
            heading_steer_gain=0.0,
            center_steer_gain=1.0,
            yaw_damping_gain=0.0,
            straight_lookahead_weights=(0.0, 0.0, 0.0),
            corner_lookahead_weights=(0.0, 0.0, 0.0),
            racing_line_offset_ratio=apex,
            racing_line_entry_offset_ratio=entry,
            racing_line_exit_offset_ratio=exit,
            phase_aware_racing_line=True,
            wall_balance_gain=0.0,
            normal_steer_limit=1.0,
            steer_slew_per_tick=1.0,
        )
    )


def _v2_line_controller(
    *,
    apex: float = 0.0,
    entry: float = 0.0,
    exit: float = 0.0,
    center_steer_gain: float = 1.0,
    line_target_slew_per_tick: float = 1.0,
    line_clearance_m: float = 0.0,
    maximum_racing_line_offset_ratio: float = MAX_RACING_LINE_OFFSET_RATIO,
    line_target_release_per_tick: float | None = None,
    sweeper_minimum_duration_s: float = 0.0,
    sweeper_speed_hold_seconds: float = 0.0,
    sweeper_target_speed_bonus_mps: float = 0.0,
    initialize_line_target_from_preview: bool = False,
) -> PreviewController:
    return PreviewController(
        ControllerParameters(
            heading_steer_gain=0.0,
            center_steer_gain=center_steer_gain,
            yaw_damping_gain=0.0,
            straight_lookahead_weights=(0.0, 0.0, 0.0),
            corner_lookahead_weights=(0.0, 0.0, 0.0),
            racing_line_offset_ratio=apex,
            racing_line_entry_offset_ratio=entry,
            racing_line_exit_offset_ratio=exit,
            phase_aware_racing_line=True,
            pose_invariant_racing_line=True,
            curvature_offset_compensation=1.0,
            curvature_heading_compensation=1.0,
            line_turn_sensitivity=0.03,
            maximum_racing_line_offset_ratio=maximum_racing_line_offset_ratio,
            line_target_slew_per_tick=line_target_slew_per_tick,
            line_target_release_per_tick=line_target_release_per_tick,
            sweeper_minimum_duration_s=sweeper_minimum_duration_s,
            sweeper_speed_hold_seconds=sweeper_speed_hold_seconds,
            sweeper_target_speed_bonus_mps=sweeper_target_speed_bonus_mps,
            initialize_line_target_from_preview=initialize_line_target_from_preview,
            line_clearance_m=line_clearance_m,
            wall_balance_gain=0.0,
            normal_steer_limit=1.0,
            steer_slew_per_tick=1.0,
        )
    )


def _centreline_sensors(
    model: TrackProgressModel,
    progress_m: float,
    *,
    lateral_m: float = 0.0,
    yaw_degrees: float = 0.0,
) -> RobotSensors:
    """Build the exact camera reading for a car posed against the real track."""
    pose = track_pose_at_distance(model, progress_m)
    track_radians = radians(pose.heading_degrees)
    right_x, right_z = cos(track_radians), -sin(track_radians)
    origin_x = pose.position.x + right_x * lateral_m
    origin_z = pose.position.z + right_z * lateral_m
    car_radians = radians(pose.heading_degrees + yaw_degrees)
    car_right_x, car_right_z = cos(car_radians), -sin(car_radians)

    def lateral_to(x: float, z: float) -> float:
        return (x - origin_x) * car_right_x + (z - origin_z) * car_right_z

    return RobotSensors(
        camera=CameraSensors(
            center_offset_m=lateral_to(pose.position.x, pose.position.z),
            heading_error_degrees=-yaw_degrees,
            lookahead_offsets_m=tuple(
                lateral_to(*_point_xz(track_pose_at_distance(model, progress_m + distance_m).position))
                for distance_m in TRACK_LOOKAHEAD_DISTANCES_M
            ),
            lookahead_distances_m=TRACK_LOOKAHEAD_DISTANCES_M,
        )
    )


def _point_xz(point: TrackPoint) -> tuple[float, float]:
    return point.x, point.z


def _phase_mass_over_one_lap(**pose: float) -> dict[str, float]:
    """Average entry/apex/exit weight over every corner metre of the real track."""
    model = track_progress_model_for_layout(TRACK_ID_MUGELLO_SHORT)
    parameters = ControllerParameters()
    totals = {"entry": 0.0, "apex": 0.0, "exit": 0.0}
    corner_samples = 0
    sample_count = int(model.total_length_m / 0.5)
    for index in range(sample_count):
        features = build_preview_features(_centreline_sensors(model, index * 0.5, **pose), parameters)
        near_turn, far_turn = track_shape_preview(features, parameters)
        dominant = max(abs(near_turn), abs(far_turn))
        if dominant <= 0.02:
            continue
        corner_samples += 1
        phase_difference = (abs(far_turn) - abs(near_turn)) / dominant
        entry = min(1.0, max(0.0, phase_difference / LINE_PHASE_TRANSITION_RATIO))
        exit_ = min(1.0, max(0.0, -phase_difference / LINE_PHASE_TRANSITION_RATIO))
        totals["entry"] += entry
        totals["exit"] += exit_
        totals["apex"] += max(0.0, 1.0 - entry - exit_)
    assert corner_samples > 100
    return {phase: total / corner_samples for phase, total in totals.items()}


def test_phase_selector_does_not_pin_to_entry_over_a_whole_lap() -> None:
    """Falsify the defect that made the first CEM gate meaningless.

    Reading ``c_i / d_i`` as curvature put 85% of the weight on entry and 8% on
    apex over this same lap, so the line asked the car to sit outside for the
    whole corner and the optimizer's only sane reply was to switch it off.
    """
    mass = _phase_mass_over_one_lap()

    assert mass["entry"] < 0.55, mass
    assert mass["exit"] < 0.70, mass
    # Apex is the smallest share by construction: entry and exit each saturate
    # over a whole tail of the phase difference, while apex only holds the band
    # between them, and this track's corners are short against a 16 m preview.
    assert mass["apex"] > 0.08, mass


def test_phase_mass_is_unchanged_when_the_car_is_displaced_and_yawed() -> None:
    on_line = _phase_mass_over_one_lap()
    displaced = _phase_mass_over_one_lap(lateral_m=2.0, yaw_degrees=10.0)

    for phase, share in on_line.items():
        assert abs(displaced[phase] - share) < 0.05, (phase, share, displaced[phase])


def test_track_shape_preview_cancels_lateral_offset_exactly() -> None:
    """Displacing the car sideways must not move the signal at all.

    Both differences are taken between measurements sharing one origin, so a
    lateral shift is common-mode and cancels algebraically rather than
    approximately.  This is the property the retired ``kappa`` signal needed
    ``curvature_offset_compensation`` to approximate, and never achieved.
    """
    model = track_progress_model_for_layout(TRACK_ID_MUGELLO_SHORT)
    parameters = ControllerParameters()
    for progress_m in (12.0, 31.5, 44.0, 57.0, 88.5, 143.0):
        aligned = track_shape_preview(
            build_preview_features(_centreline_sensors(model, progress_m), parameters), parameters
        )
        displaced = track_shape_preview(
            build_preview_features(_centreline_sensors(model, progress_m, lateral_m=1.8), parameters),
            parameters,
        )
        assert abs(displaced[0] - aligned[0]) < 1e-12, (progress_m, aligned, displaced)
        assert abs(displaced[1] - aligned[1]) < 1e-12, (progress_m, aligned, displaced)


def test_track_shape_preview_leaves_only_a_bounded_yaw_residual() -> None:
    """Yaw leaves a bounded second-order residual, not a first-order leak.

    Heading error tilts every segment slope equally and so cancels from both
    differences to first order.  What survives is the genuine geometry change:
    a yawed car's 16 m lookahead lands on a different piece of track.
    """
    model = track_progress_model_for_layout(TRACK_ID_MUGELLO_SHORT)
    parameters = replace(ControllerParameters(), curvature_offset_compensation=1.0, curvature_heading_compensation=1.0)
    worst_shape = 0.0
    for index in range(int(model.total_length_m / 0.5)):
        progress_m = index * 0.5
        aligned = build_preview_features(_centreline_sensors(model, progress_m), parameters)
        posed = build_preview_features(
            _centreline_sensors(model, progress_m, lateral_m=1.8, yaw_degrees=-9.0), parameters
        )
        worst_shape = max(
            worst_shape,
            max(
                abs(posed_value - aligned_value)
                for posed_value, aligned_value in zip(
                    track_shape_preview(posed, parameters),
                    track_shape_preview(aligned, parameters),
                    strict=True,
                )
            ),
        )

    # The signal peaks near 0.16 on this track, so this bounds the worst-case
    # drift at a quarter of full scale while the lap mean stays near a twentieth.
    assert worst_shape < 0.040, worst_shape


def test_track_shape_preview_reports_constant_curvature_through_a_steady_corner() -> None:
    """A constant-radius corner must read the same near and far, not rising."""
    radius_m = 30.0
    parameters = ControllerParameters()
    offsets_m = tuple(distance_m * distance_m / (2.0 * radius_m) for distance_m in TRACK_LOOKAHEAD_DISTANCES_M)
    features = build_preview_features(
        RobotSensors(
            camera=CameraSensors(
                lookahead_offsets_m=offsets_m,
                lookahead_distances_m=TRACK_LOOKAHEAD_DISTANCES_M,
            )
        ),
        parameters,
    )

    near_turn, far_turn = track_shape_preview(features, parameters)

    assert abs(near_turn - 1.0 / radius_m) < 1e-9
    assert abs(far_turn - 1.0 / radius_m) < 1e-9
    # The retired signal read c_i / d_i, which rises with lookahead distance and
    # therefore called this steady corner an entry.
    assert abs(far_turn - near_turn) < 1e-9


def test_line_clamp_is_a_parameter_and_defaults_to_the_legacy_constant() -> None:
    assert ControllerParameters().maximum_racing_line_offset_ratio == MAX_RACING_LINE_OFFSET_RATIO

    wide = _v2_line_controller(entry=0.90, maximum_racing_line_offset_ratio=0.85)
    narrow = _v2_line_controller(entry=0.90)
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.30, 2.20)))

    wide(sensors)
    narrow(sensors)

    assert wide.state.line_target == pytest.approx(-0.85)
    assert narrow.state.line_target == pytest.approx(-MAX_RACING_LINE_OFFSET_RATIO)


def test_pose_invariant_speed_curvature_ignores_the_car_s_own_offset() -> None:
    """A straight seen from off-line must not read as a corner.

    The default signal divides the cumulative offset by distance, so a displaced
    car manufactures curvature and the search's only reply is to raise
    `curvature_lateral_ratio` until the term is desensitised.
    """
    model = track_progress_model_for_layout(TRACK_ID_MUGELLO_SHORT)
    straight_m = _straightest_progress_m(model)
    parameters = replace(ControllerParameters(), pose_invariant_speed_curvature=True)
    legacy = ControllerParameters()

    centred = _centreline_sensors(model, straight_m)
    displaced = _centreline_sensors(model, straight_m, lateral_m=2.0)

    invariant = [
        max(abs(value) for value in track_shape_preview(build_preview_features(s, parameters), parameters))
        for s in (centred, displaced)
    ]
    legacy_values = [
        max(abs(value) for value in track_curvature_preview(build_preview_features(s, legacy), legacy))
        for s in (centred, displaced)
    ]

    assert abs(invariant[1] - invariant[0]) < 1e-12, invariant
    assert legacy_values[1] - legacy_values[0] > 0.25, legacy_values


def test_pose_invariant_speed_curvature_is_off_by_default() -> None:
    assert ControllerParameters().pose_invariant_speed_curvature is False


def _straightest_progress_m(model: TrackProgressModel) -> float:
    """Find the flattest metre of the lap, so 'straight' is not an assumption."""
    parameters = ControllerParameters()
    best_progress_m = 0.0
    best_curvature = float("inf")
    for index in range(int(model.total_length_m / 0.5)):
        progress_m = index * 0.5
        features = build_preview_features(_centreline_sensors(model, progress_m), parameters)
        curvature = max(abs(value) for value in track_shape_preview(features, parameters))
        if curvature < best_curvature:
            best_curvature, best_progress_m = curvature, progress_m
    return best_progress_m


CORNER_AHEAD = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.30, 2.20)))
STRAIGHT = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.0, 0.0)))
OPPOSITE_CORNER = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, -0.30, -2.20)))


def test_release_rate_defaults_to_the_outward_slew() -> None:
    assert ControllerParameters().line_target_release_per_tick is None

    symmetric = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.10)
    explicit = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.10, line_target_release_per_tick=0.10)
    for _ in range(10):
        symmetric(CORNER_AHEAD)
        explicit(CORNER_AHEAD)
    for _ in range(5):
        symmetric(STRAIGHT)
        explicit(STRAIGHT)

    assert symmetric.state.line_target == pytest.approx(explicit.state.line_target)


def test_a_slow_release_holds_the_line_through_a_straight() -> None:
    """The behaviour the taut in-corridor path needs and v3 could not express."""
    holding = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.20, line_target_release_per_tick=0.002)
    releasing = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.20, line_target_release_per_tick=0.20)
    for _ in range(10):
        holding(CORNER_AHEAD)
        releasing(CORNER_AHEAD)
    committed = holding.state.line_target
    assert committed < -0.30, committed

    for _ in range(20):
        holding(STRAIGHT)
        releasing(STRAIGHT)

    # The releasing controller is dragged back to the centreline; the holding one
    # keeps almost all of the offset it worked for.
    assert releasing.state.line_target == pytest.approx(0.0, abs=1e-9)
    assert holding.state.line_target < committed * 0.85, holding.state.line_target


def test_release_rate_never_slows_a_crossing_to_the_other_side() -> None:
    """An opposite corner must still be answered at the fast outward rate."""
    controller = _v2_line_controller(entry=0.65, line_target_slew_per_tick=0.20, line_target_release_per_tick=0.002)
    for _ in range(10):
        controller(CORNER_AHEAD)
    committed = controller.state.line_target
    assert committed < -0.30, committed

    controller(OPPOSITE_CORNER)

    # One tick of the fast rate, not the slow one.
    moved = controller.state.line_target - committed
    assert moved == pytest.approx(0.20, abs=1e-9), moved


def test_sweeper_speed_bonus_is_default_off() -> None:
    controller = _v2_line_controller(apex=0.4, entry=0.4)
    sustained_left = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(-0.5, -2.0, -5.0)))

    for _ in range(120):
        controller(sustained_left)

    assert controller.state.sweeper_turn_seconds == 0.0
    assert controller.state.sweeper_speed_hold_seconds == 0.0


def test_sweeper_speed_bonus_holds_through_straight_and_cancels_on_opposite_turn() -> None:
    controller = _v2_line_controller(
        apex=0.4,
        entry=0.4,
        line_target_slew_per_tick=1.0,
        sweeper_minimum_duration_s=0.05,
        sweeper_speed_hold_seconds=0.20,
        sweeper_target_speed_bonus_mps=1.0,
    )
    sustained_left = RobotSensors(
        dt_s=1.0 / 60.0,
        camera=CameraSensors(lookahead_offsets_m=(-0.5, -2.0, -5.0)),
    )
    sustained_right = RobotSensors(
        dt_s=1.0 / 60.0,
        camera=CameraSensors(lookahead_offsets_m=(0.5, 2.0, 5.0)),
    )
    straight = RobotSensors(dt_s=1.0 / 60.0)

    for _ in range(4):
        controller(sustained_left)

    assert controller.state.sweeper_speed_hold_direction == -1.0
    assert controller.state.sweeper_speed_hold_seconds > 0.0

    for _ in range(4):
        controller(straight)

    assert controller.state.sweeper_speed_hold_seconds > 0.0

    controller(sustained_right)

    assert controller.state.sweeper_speed_hold_seconds == 0.0
    assert controller.state.sweeper_speed_hold_direction == 0.0


def test_sweeper_speed_bonus_activates_only_after_sustained_turn() -> None:
    parameters = ControllerParameters(
        pose_invariant_racing_line=True,
        straight_target_speed_mps=10.0,
        corner_target_speed_mps=10.0,
        steering_speed_reduction=0.0,
        throttle_gain=1.0,
        sweeper_minimum_duration_s=0.05,
        sweeper_speed_hold_seconds=0.20,
        sweeper_target_speed_bonus_mps=2.0,
    )
    controller = PreviewController(parameters)
    sustained_left = RobotSensors(
        dt_s=1.0 / 60.0,
        odometry=OdometrySensors(speed_mps=10.5),
        camera=CameraSensors(lookahead_offsets_m=(-0.5, -2.0, -5.0)),
    )

    assert controller(sustained_left).throttle == 0.0
    assert controller(sustained_left).throttle == 0.0
    assert controller(sustained_left).throttle > 0.0


def test_sweeper_preview_bonus_arms_on_broad_entry_band() -> None:
    parameters = ControllerParameters(
        pose_invariant_racing_line=True,
        straight_target_speed_mps=10.0,
        corner_target_speed_mps=10.0,
        steering_speed_reduction=0.0,
        throttle_gain=1.0,
        sweeper_preview_minimum_far_curvature=0.10,
        sweeper_preview_maximum_far_curvature=0.14,
        sweeper_preview_speed_hold_seconds=0.20,
        sweeper_preview_target_speed_bonus_mps=2.0,
    )
    controller = PreviewController(parameters)
    broad_entry = RobotSensors(
        dt_s=1.0 / 60.0,
        odometry=OdometrySensors(speed_mps=10.5),
        camera=CameraSensors(lookahead_offsets_m=(0.4, 1.175, -2.95)),
    )

    command = controller(broad_entry)

    assert controller.state.sweeper_preview_speed_hold_seconds == 0.20
    assert controller.state.sweeper_preview_hold_direction == -1.0
    assert command.throttle > 0.0


def test_sweeper_preview_bonus_rejects_tighter_entry_and_cancels_on_opposite_turn() -> None:
    parameters = ControllerParameters(
        pose_invariant_racing_line=True,
        straight_target_speed_mps=10.0,
        corner_target_speed_mps=10.0,
        steering_speed_reduction=0.0,
        throttle_gain=1.0,
        sweeper_preview_minimum_far_curvature=0.10,
        sweeper_preview_maximum_far_curvature=0.14,
        sweeper_preview_speed_hold_seconds=0.20,
        sweeper_preview_target_speed_bonus_mps=2.0,
    )
    controller = PreviewController(parameters)
    broad_entry = RobotSensors(
        dt_s=1.0 / 60.0,
        odometry=OdometrySensors(speed_mps=10.5),
        camera=CameraSensors(lookahead_offsets_m=(0.4, 1.175, -2.95)),
    )
    tight_entry = RobotSensors(
        dt_s=1.0 / 60.0,
        odometry=OdometrySensors(speed_mps=10.5),
        camera=CameraSensors(lookahead_offsets_m=(0.4, 1.175, -4.88)),
    )
    opposite_turn = RobotSensors(
        dt_s=1.0 / 60.0,
        odometry=OdometrySensors(speed_mps=10.5),
        camera=CameraSensors(lookahead_offsets_m=(0.5, 2.0, 5.0)),
    )

    assert controller(tight_entry).throttle == 0.0
    assert controller.state.sweeper_preview_speed_hold_seconds == 0.0

    controller(broad_entry)
    controller(opposite_turn)

    assert controller.state.sweeper_preview_speed_hold_seconds == 0.0
    assert controller.state.sweeper_preview_hold_direction == 0.0


def test_corner_exit_bonus_tracks_pose_invariant_exit_phase() -> None:
    parameters = ControllerParameters(
        straight_target_speed_mps=10.0,
        corner_target_speed_mps=10.0,
        steering_speed_reduction=0.0,
        throttle_gain=1.0,
        corner_exit_target_speed_bonus_mps=2.0,
    )
    controller = PreviewController(parameters)

    controller._update_corner_exit_speed_state((-0.12, -0.02))  # pyright: ignore[reportPrivateUsage]
    features = build_preview_features(
        RobotSensors(odometry=OdometrySensors(speed_mps=10.5)),
        parameters,
    )
    throttle, target_speed = controller._speed_command(  # pyright: ignore[reportPrivateUsage]
        features,
        curvature=0.0,
        steer=0.0,
        avoiding=False,
        startup_speed_cap_mps=None,
    )

    assert controller.state.corner_exit_speed_bonus_factor == 1.0
    assert target_speed == 12.0
    assert throttle > 0.0

    controller._update_corner_exit_speed_state((-0.02, -0.12))  # pyright: ignore[reportPrivateUsage]

    assert controller.state.corner_exit_speed_bonus_factor == 0.0


def test_line_target_can_initialize_from_the_first_preview() -> None:
    cold = _v2_line_controller(
        apex=0.8,
        entry=0.8,
        exit=0.8,
        line_target_slew_per_tick=0.01,
    )
    warm = _v2_line_controller(
        apex=0.8,
        entry=0.8,
        exit=0.8,
        line_target_slew_per_tick=0.01,
        initialize_line_target_from_preview=True,
    )
    corner = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.5, 2.0, 5.0)))

    cold(corner)
    warm(corner)

    assert abs(cold.state.line_target) == 0.01
    assert abs(warm.state.line_target) > abs(cold.state.line_target)
    assert warm.state.line_target_initialized
