#!/usr/bin/env python3
"""Record one preview-controller trial as strict per-tick JSON Lines."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import TextIO, cast

from scripts.controller_training.evaluator import SOLO_TRIAL_DEFAULT_SECONDS, SoloEvaluator, SoloTrialResult

from controllers.preview_controller import (
    LINE_PHASE_TRANSITION_RATIO,
    ControllerParameters,
    ControlMode,
    PreviewController,
    PreviewDiagnostics,
)
from racing import RobotCommand, RobotSensors

TRACE_SCHEMA_VERSION = 1
STRONG_LINE_TARGET_M = 0.8
# Local curvature (1/m) above which the phase selector is reading a real corner
# rather than straight-line noise.  Phase mass is only meaningful over these ticks.
PHASE_ACTIVE_CURVATURE = 0.02


def _phase_weights(shape: tuple[float, float]) -> tuple[float, float, float]:
    """Mirror the controller's entry/apex/exit split for offline telemetry.

    A selector that reports entry almost everywhere is the failure mode that
    silently wasted the first CEM gate, so the mass split is recorded per run.
    """
    near_magnitude, far_magnitude = abs(shape[0]), abs(shape[1])
    dominant = max(near_magnitude, far_magnitude)
    if dominant <= 1e-6:
        return 0.0, 0.0, 0.0
    phase_difference = (far_magnitude - near_magnitude) / dominant
    entry = min(1.0, max(0.0, phase_difference / LINE_PHASE_TRANSITION_RATIO))
    exit_ = min(1.0, max(0.0, -phase_difference / LINE_PHASE_TRANSITION_RATIO))
    return entry, max(0.0, 1.0 - entry - exit_), exit_


@dataclass(slots=True)
class TraceCollector:
    """Controller wrapper that records the exact pre-action observation."""

    controller: PreviewController
    stream: TextIO
    diagnostics: PreviewDiagnostics | None = None
    tick_count: int = 0
    straight_tick_count: int = 0
    corner_tick_count: int = 0
    avoid_tick_count: int = 0
    straight_absolute_offset_sum_m: float = 0.0
    straight_absolute_line_target_sum_m: float = 0.0
    straight_absolute_center_line_steer_sum: float = 0.0
    straight_absolute_wall_steer_sum: float = 0.0
    straight_tracking_error_sum_m: float = 0.0
    straight_curvature_sum: float = 0.0
    straight_target_speed_sum_mps: float = 0.0
    straight_full_throttle_tick_count: int = 0
    corner_speed_sum_mps: float = 0.0
    maximum_yaw_rate_per_speed: float = 0.0
    strong_line_target_tick_count: int = 0
    strong_line_target_absolute_target_sum_m: float = 0.0
    strong_line_target_directional_offset_sum_m: float = 0.0
    strong_line_target_tracking_error_sum_m: float = 0.0
    strong_line_target_opposite_side_tick_count: int = 0
    phase_tick_count: int = 0
    phase_entry_weight_sum: float = 0.0
    phase_apex_weight_sum: float = 0.0
    phase_exit_weight_sum: float = 0.0

    def __post_init__(self) -> None:
        self.controller.diagnostics_sink = self._capture_diagnostics

    def _capture_diagnostics(self, diagnostics: PreviewDiagnostics) -> None:
        self.diagnostics = diagnostics

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        self.diagnostics = None
        command = self.controller(sensors)
        diagnostics = self.diagnostics
        if diagnostics is None:
            raise RuntimeError("preview diagnostics were not captured")

        parameters = self.controller.parameters
        signed_offset_m = -sensors.camera.center_offset_m
        line_target_m = diagnostics.line_target * parameters.center_offset_cap_m
        left_wall_m = sensors.wall_lidar.left_m
        right_wall_m = sensors.wall_lidar.right_m
        entry_weight, apex_weight, exit_weight = _phase_weights(diagnostics.line_shape)
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "record_type": "controller_trace_tick",
            "tick": sensors.tick,
            "time_s": sensors.tick * sensors.dt_s,
            "mode": diagnostics.mode.value,
            "speed_mps": sensors.odometry.speed_mps,
            "yaw_rate_degrees_per_s": sensors.imu.yaw_rate_degrees_per_s,
            "yaw_rate_per_speed": abs(sensors.imu.yaw_rate_degrees_per_s) / max(abs(sensors.odometry.speed_mps), 1e-6),
            "throttle": command.throttle,
            "steer": command.steer,
            "desired_steer": diagnostics.desired_steer,
            "target_speed_mps": diagnostics.target_speed_mps,
            "curvature": diagnostics.curvature,
            "kappa_0": diagnostics.kappa[0],
            "kappa_1": diagnostics.kappa[1],
            "kappa_2": diagnostics.kappa[2],
            "shape_near": diagnostics.line_shape[0],
            "shape_far": diagnostics.line_shape[1],
            "phase_entry_weight": entry_weight,
            "phase_apex_weight": apex_weight,
            "phase_exit_weight": exit_weight,
            "line_target": diagnostics.line_target,
            "line_target_m": line_target_m,
            "signed_lateral_offset_m": signed_offset_m,
            "line_tracking_error_m": abs(line_target_m - signed_offset_m),
            "heading_steer": diagnostics.heading_steer,
            "center_line_steer": diagnostics.center_line_steer,
            "preview_steer": diagnostics.preview_steer,
            "yaw_damping_steer": diagnostics.yaw_damping_steer,
            "wall_balance_steer": diagnostics.wall_balance_steer,
            "wall_left_m": _json_number(left_wall_m),
            "wall_right_m": _json_number(right_wall_m),
            "wall_near_m": _json_number(min(left_wall_m, right_wall_m)),
            "wall_far_m": _json_number(max(left_wall_m, right_wall_m)),
            "wall_contact_s": sensors.contact.wall,
            "damage": sensors.contact.damage,
        }
        _write_record(self.stream, record)
        self.tick_count += 1
        yaw_rate_per_speed = abs(sensors.imu.yaw_rate_degrees_per_s) / max(abs(sensors.odometry.speed_mps), 1e-6)
        self.maximum_yaw_rate_per_speed = max(self.maximum_yaw_rate_per_speed, yaw_rate_per_speed)
        if diagnostics.mode is ControlMode.AVOID:
            self.avoid_tick_count += 1
        if abs(line_target_m) >= STRONG_LINE_TARGET_M:
            self.strong_line_target_tick_count += 1
            self.strong_line_target_absolute_target_sum_m += abs(line_target_m)
            self.strong_line_target_directional_offset_sum_m += (
                signed_offset_m if line_target_m > 0.0 else -signed_offset_m
            )
            self.strong_line_target_tracking_error_sum_m += abs(line_target_m - signed_offset_m)
            if line_target_m * signed_offset_m < 0.0:
                self.strong_line_target_opposite_side_tick_count += 1
        if max(abs(value) for value in diagnostics.line_shape) > PHASE_ACTIVE_CURVATURE:
            self.phase_tick_count += 1
            self.phase_entry_weight_sum += entry_weight
            self.phase_apex_weight_sum += apex_weight
            self.phase_exit_weight_sum += exit_weight
        if diagnostics.curvature < 0.15:
            self.straight_tick_count += 1
            self.straight_absolute_offset_sum_m += abs(signed_offset_m)
            self.straight_absolute_line_target_sum_m += abs(line_target_m)
            self.straight_absolute_center_line_steer_sum += abs(diagnostics.center_line_steer)
            self.straight_absolute_wall_steer_sum += abs(diagnostics.wall_balance_steer)
            self.straight_tracking_error_sum_m += abs(line_target_m - signed_offset_m)
            self.straight_curvature_sum += diagnostics.curvature
            self.straight_target_speed_sum_mps += diagnostics.target_speed_mps
            if command.throttle >= parameters.maximum_forward_throttle - 1e-9:
                self.straight_full_throttle_tick_count += 1
        if diagnostics.curvature > 0.55:
            self.corner_tick_count += 1
            self.corner_speed_sum_mps += sensors.odometry.speed_mps
        return command

    def write_summary(self, result: SoloTrialResult) -> None:
        straight_count = max(1, self.straight_tick_count)
        strong_target_count = max(1, self.strong_line_target_tick_count)
        _write_record(
            self.stream,
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "record_type": "controller_trace_summary",
                "tick_count": self.tick_count,
                "straight_tick_count": self.straight_tick_count,
                "corner_tick_count": self.corner_tick_count,
                "avoid_tick_count": self.avoid_tick_count,
                "straight_mean_absolute_offset_m": self.straight_absolute_offset_sum_m / straight_count,
                "straight_mean_absolute_line_target_m": self.straight_absolute_line_target_sum_m / straight_count,
                "straight_mean_absolute_center_line_steer": (
                    self.straight_absolute_center_line_steer_sum / straight_count
                ),
                "straight_mean_absolute_wall_steer": self.straight_absolute_wall_steer_sum / straight_count,
                "straight_mean_tracking_error_m": self.straight_tracking_error_sum_m / straight_count,
                "straight_mean_curvature": self.straight_curvature_sum / straight_count,
                "straight_mean_target_speed_mps": self.straight_target_speed_sum_mps / straight_count,
                "straight_full_throttle_share": self.straight_full_throttle_tick_count / straight_count,
                "corner_mean_speed_mps": self.corner_speed_sum_mps / max(1, self.corner_tick_count),
                "phase_tick_count": self.phase_tick_count,
                "phase_entry_mass": self.phase_entry_weight_sum / max(1, self.phase_tick_count),
                "phase_apex_mass": self.phase_apex_weight_sum / max(1, self.phase_tick_count),
                "phase_exit_mass": self.phase_exit_weight_sum / max(1, self.phase_tick_count),
                "maximum_yaw_rate_per_speed": self.maximum_yaw_rate_per_speed,
                "strong_line_target_threshold_m": STRONG_LINE_TARGET_M,
                "strong_line_target_tick_count": self.strong_line_target_tick_count,
                "strong_line_target_mean_absolute_target_m": (
                    self.strong_line_target_absolute_target_sum_m / strong_target_count
                ),
                "strong_line_target_mean_directional_offset_m": (
                    self.strong_line_target_directional_offset_sum_m / strong_target_count
                ),
                "strong_line_target_mean_tracking_error_m": (
                    self.strong_line_target_tracking_error_sum_m / strong_target_count
                ),
                "strong_line_target_opposite_side_tick_count": self.strong_line_target_opposite_side_tick_count,
                "trial": result.to_dict(),
            },
        )


def _json_number(value: float) -> float | None:
    return value if isfinite(value) else None


def _write_record(stream: TextIO, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _parameter_overrides(values: Sequence[str], base: ControllerParameters) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not hasattr(base, name):
            raise ValueError(f"expected an existing ControllerParameters field as NAME=VALUE, got {value!r}")
        existing = getattr(base, name)
        if isinstance(existing, bool):
            normalized = raw.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"boolean override {name!r} must be true or false")
            overrides[name] = normalized == "true"
        elif isinstance(existing, float):
            parsed = float(raw)
            if not isfinite(parsed):
                raise ValueError(f"numeric override {name!r} must be finite")
            overrides[name] = parsed
        else:
            raise ValueError(f"trace overrides do not support field {name!r}")
    return overrides


def _controller(module: str, preset: str | None, overrides: Sequence[str]) -> PreviewController:
    if preset is None:
        loaded_module = import_module(module)
        raw_factory = getattr(loaded_module, "create_controller", None)
        if not callable(raw_factory):
            raise TypeError(f"trace module {module!r} must define create_controller()")
        factory = cast(Callable[[], object], raw_factory)
        loaded = factory()
        if not isinstance(loaded, PreviewController):
            raise TypeError("trace requires a controller created by controllers.preview_controller.PreviewController")
        base = loaded.parameters
    else:
        from scripts.controller_training.search import preset_configuration

        base, _ = preset_configuration(preset)
    return PreviewController(replace(base, **_parameter_overrides(overrides, base)))


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module",
        nargs="?",
        default="controllers.race_faster",
        help="dotted preview-controller module (default: controllers.race_faster)",
    )
    parser.add_argument("--preset")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--seconds", type=float, default=SOLO_TRIAL_DEFAULT_SECONDS)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    output = cast(Path | None, args.output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    stream = sys.stdout if output is None else output.open("w", encoding="utf-8")
    try:
        controller = _controller(str(args.module), cast(str | None, args.preset), cast(list[str], args.overrides))
        collector = TraceCollector(controller=controller, stream=stream)
        with SoloEvaluator() as evaluator:
            result = evaluator.run_trial(
                controller_factory=lambda: collector,
                seed=int(args.seed),
                duration_seconds=float(args.seconds),
            )
        collector.write_summary(result)
    finally:
        if output is not None:
            stream.close()


if __name__ == "__main__":
    main()
