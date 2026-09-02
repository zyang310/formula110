"""Progress-aware hybrid Formula 110 controller using only legal sensors."""

from __future__ import annotations

from math import isfinite

from controllers.hybrid_track_policy_data import DEFAULT_GAINS, POLICY_ROWS, TRACK_TOTAL_LENGTH_M, TRACK_WIDTH_M
from racing import RobotCommand, RobotSensors
from racing.track.world import clamp

RACING_NAME: str = "Hybrid Track Policy"
RACING_COLOR: str = "#2dd4bf"

PolicyRow = tuple[float, float, float, float, float, float, float, float, float]
_global_controller: HybridTrackController | None = None


class HybridTrackController:
    """Stateful legal-sensor controller with progress-indexed policy lookup."""

    def __init__(self, overrides: dict[str, float] | None = None) -> None:
        self._gains = dict(DEFAULT_GAINS)
        if overrides:
            self._gains.update(overrides)
        self._progress_m: float | None = None
        self._previous_odometry_m = 0.0
        self._previous_damage = 0.0
        self._slow_seconds = 0.0
        self._recovery_seconds = 0.0
        self._recovery_phase = 0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        dt_s = max(0.0, sensors.dt_s)
        speed_mps = sensors.odometry.speed_mps
        self._update_progress_estimate(sensors)
        row = self._policy_row()
        target_speed_mps = self._bounded_target_speed(row, sensors)
        target_offset_m = row[7]
        command = self._normal_command(sensors, row, target_speed_mps, target_offset_m)
        command = self._apply_emergencies(sensors, command, target_speed_mps)
        if abs(speed_mps) < self._gains["stuck_speed_mps"] and sensors.tick > 18:
            self._slow_seconds += dt_s
        else:
            self._slow_seconds = max(0.0, self._slow_seconds - 2.5 * dt_s)
        if self._slow_seconds > self._gains["stuck_seconds"] or sensors.contact.damage > self._previous_damage + 0.03:
            self._recovery_seconds = max(self._recovery_seconds, 0.75)
            self._recovery_phase += 1
        self._previous_damage = sensors.contact.damage
        if self._recovery_seconds > 0.0:
            self._recovery_seconds = max(0.0, self._recovery_seconds - dt_s)
            command = self._recovery_command(sensors)
        return RobotCommand(throttle=clamp(command.throttle, -1.0, 1.0), steer=clamp(command.steer, -1.0, 1.0))

    def _update_progress_estimate(self, sensors: RobotSensors) -> None:
        observed_progress = self._signature_progress(sensors)
        distance_delta = sensors.odometry.distance_m - self._previous_odometry_m
        self._previous_odometry_m = sensors.odometry.distance_m
        if self._progress_m is None:
            self._progress_m = observed_progress
            return
        heading_factor = max(0.0, 1.0 - abs(sensors.camera.heading_error_degrees) / 115.0)
        predicted = (self._progress_m + max(-3.0, min(3.0, distance_delta * heading_factor))) % TRACK_TOTAL_LENGTH_M
        observation_error = _circular_distance(observed_progress, predicted)
        observation_weight = 0.18 if observation_error < 14.0 else 0.45
        self._progress_m = _circular_blend(predicted, observed_progress, observation_weight)

    def _signature_progress(self, sensors: RobotSensors) -> float:
        if not sensors.camera.visible:
            return 0.0 if self._progress_m is None else self._progress_m
        track_heading = _wrap_degrees(sensors.imu.heading_degrees + sensors.camera.heading_error_degrees)
        observed_shape = _lookahead_shape(sensors)
        best_score = float("inf")
        best_progress = 0.0
        predicted = self._progress_m
        for row in POLICY_ROWS:
            heading_score = abs(_heading_error(track_heading, row[1])) / 22.0
            shape_score = (
                abs(observed_shape[0] - row[3]) * 0.20
                + abs(observed_shape[1] - row[4]) * 0.13
                + abs(observed_shape[2] - row[5]) * 0.08
            )
            continuity_score = 0.0 if predicted is None else _circular_distance(row[0], predicted) / 55.0
            score = heading_score + shape_score + continuity_score
            if score < best_score:
                best_score = score
                best_progress = row[0]
        return best_progress

    def _policy_row(self) -> PolicyRow:
        if not POLICY_ROWS:
            raise RuntimeError("hybrid policy data is empty")
        progress = 0.0 if self._progress_m is None else self._progress_m % TRACK_TOTAL_LENGTH_M
        row_index = min(len(POLICY_ROWS) - 1, int(progress / TRACK_TOTAL_LENGTH_M * len(POLICY_ROWS)))
        return POLICY_ROWS[row_index]

    def _bounded_target_speed(self, row: PolicyRow, sensors: RobotSensors) -> float:
        abs_curvature = abs(row[2])
        target = row[6] * self._gains["target_speed_scale"] + self._gains["target_speed_bias_mps"]
        target -= abs_curvature * self._gains["curve_speed_penalty_scale"]
        if abs_curvature < self._gains["straight_curvature_threshold"]:
            target += self._gains["straight_speed_bonus_mps"]
        target = clamp(target, self._gains["min_target_speed_mps"], self._gains["max_target_speed_mps"])
        front_wall = sensors.wall_lidar.front_m
        if isfinite(front_wall):
            target = min(target, max(2.0, front_wall * self._gains["front_wall_speed_scale"]))
        if abs(sensors.camera.heading_error_degrees) > self._gains["large_heading_error_degrees"]:
            target = min(target, self._gains["large_heading_target_speed_mps"])
        return target

    def _normal_command(
        self,
        sensors: RobotSensors,
        row: PolicyRow,
        target_speed_mps: float,
        target_offset_m: float,
    ) -> RobotCommand:
        lookahead_offsets = sensors.camera.lookahead_offsets_m
        far_offset = lookahead_offsets[-1] if lookahead_offsets else sensors.camera.center_offset_m
        lateral_error_m = sensors.camera.center_offset_m - target_offset_m
        steer = (
            row[8]
            + sensors.camera.heading_error_degrees * self._gains["heading_gain"]
            + lateral_error_m * self._gains["center_gain"]
            + far_offset * self._gains["lookahead_gain"]
            - sensors.imu.yaw_rate_degrees_per_s * self._gains["yaw_damping_gain"]
        )
        side_bias = _finite_distance(sensors.wall_lidar.right_m, 12.0) - _finite_distance(
            sensors.wall_lidar.left_m,
            12.0,
        )
        steer += side_bias * self._gains["side_wall_bias_gain"] / max(TRACK_WIDTH_M, 1.0)
        speed_error = target_speed_mps - sensors.odometry.speed_mps
        if speed_error < -self._gains["hard_brake_speed_error"]:
            throttle = -1.0
        elif speed_error < 0.0:
            throttle = max(-0.70, speed_error * self._gains["speed_brake_kp"])
        else:
            throttle = min(1.0, 0.10 + speed_error * self._gains["speed_kp"])
        return RobotCommand(throttle=throttle, steer=steer)

    def _apply_emergencies(
        self,
        sensors: RobotSensors,
        command: RobotCommand,
        target_speed_mps: float,
    ) -> RobotCommand:
        front_wall = sensors.wall_lidar.front_m
        if isfinite(front_wall) and front_wall < self._gains["front_wall_emergency_m"]:
            steer = -0.95 if sensors.wall_lidar.left_m > sensors.wall_lidar.right_m else 0.95
            return RobotCommand(throttle=-1.0, steer=steer)
        if abs(sensors.camera.heading_error_degrees) > 65.0:
            return RobotCommand(throttle=min(command.throttle, 0.08), steer=clamp(command.steer, -0.85, 0.85))
        if sensors.odometry.speed_mps > target_speed_mps + 4.0 and abs(command.steer) > 0.65:
            return RobotCommand(throttle=min(command.throttle, -0.30), steer=command.steer)
        return command

    def _recovery_command(self, sensors: RobotSensors) -> RobotCommand:
        open_left = sensors.wall_lidar.left_m >= sensors.wall_lidar.right_m
        steer = -0.75 if open_left else 0.75
        if self._recovery_phase % 2 == 0:
            steer = -steer
        if sensors.contact.wall > 0.0 or sensors.wall_lidar.front_m < 1.8:
            return RobotCommand(throttle=-0.75, steer=steer)
        return RobotCommand(throttle=0.35, steer=-steer * 0.7)


def create_controller() -> HybridTrackController:
    return HybridTrackController()


def control(sensors: RobotSensors) -> RobotCommand:
    global _global_controller
    if _global_controller is None:
        _global_controller = create_controller()
    return _global_controller(sensors)


def _lookahead_shape(sensors: RobotSensors) -> tuple[float, float, float]:
    offsets = sensors.camera.lookahead_offsets_m
    center = sensors.camera.center_offset_m
    values = [offset - center for offset in offsets[:3]]
    while len(values) < 3:
        values.append(0.0)
    return (values[0], values[1], values[2])


def _finite_distance(value: float, fallback: float) -> float:
    return value if isfinite(value) else fallback


def _wrap_degrees(value: float) -> float:
    return value % 360.0


def _heading_error(current_heading_degrees: float, target_heading_degrees: float) -> float:
    return ((target_heading_degrees - current_heading_degrees + 180.0) % 360.0) - 180.0


def _circular_distance(a_m: float, b_m: float) -> float:
    delta = abs((a_m - b_m) % TRACK_TOTAL_LENGTH_M)
    return min(delta, TRACK_TOTAL_LENGTH_M - delta)


def _circular_blend(a_m: float, b_m: float, b_weight: float) -> float:
    delta = ((b_m - a_m + TRACK_TOTAL_LENGTH_M / 2.0) % TRACK_TOTAL_LENGTH_M) - TRACK_TOTAL_LENGTH_M / 2.0
    return (a_m + delta * clamp(b_weight, 0.0, 1.0)) % TRACK_TOTAL_LENGTH_M
