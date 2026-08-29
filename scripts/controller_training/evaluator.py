#!/usr/bin/env python3
"""Run deterministic solo controller trials with Gradescope-compatible rules."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
)
from racing.race.progress import default_track_progress_model, project_track_position
from racing.race.runtime import (
    RaceCarRuntime,
    lap_progress_tracker_for_spawn_pose,
    race_contact_states,
    race_spawn_poses,
    robot_is_eliminated,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import build_robot_sensors
from racing.student.api import RobotCommand, RobotController, RobotSensors, load_student_controller

SOLO_TRIAL_RESULT_SCHEMA_VERSION = 1
SOLO_TRIAL_RESULT_RECORD_TYPE = "solo_trial_result"
SOLO_TRIAL_DEFAULT_SECONDS = 30.0
SOLO_TRIAL_FIXED_DELTA_SECONDS = 1.0 / 60.0

# Candidate parameters can be closed over by this zero-argument factory. Calling
# it once per trial prevents smoothing and recovery state from leaking between
# seeds while keeping the evaluator independent of a particular policy type.
ParameterizedControllerFactory = Callable[[], RobotController]


@dataclass(frozen=True, slots=True)
class SoloTrialResult:
    """Stable metrics produced by one marshal-free solo trial."""

    seed: int
    elapsed_seconds: float
    raw_distance_m: float
    partial_laps: float
    lap_count: int
    damage: float
    survived: bool
    wall_contact_seconds: float
    max_speed_mps: float
    first_lap_time_seconds: float | None
    best_lap_time_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the versioned, JSON-compatible trial record."""
        return {
            "schema_version": SOLO_TRIAL_RESULT_SCHEMA_VERSION,
            "record_type": SOLO_TRIAL_RESULT_RECORD_TYPE,
            "ok": True,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "raw_distance_m": self.raw_distance_m,
            "partial_laps": self.partial_laps,
            "lap_count": self.lap_count,
            "damage": self.damage,
            "survived": self.survived,
            "wall_contact_seconds": self.wall_contact_seconds,
            "max_speed_mps": self.max_speed_mps,
            "first_lap_time_seconds": self.first_lap_time_seconds,
            "best_lap_time_seconds": self.best_lap_time_seconds,
        }


class SoloEvaluator:
    """Reusable headless scene for deterministic one-car evaluations."""

    def __init__(self, *, fixed_delta_seconds: float = SOLO_TRIAL_FIXED_DELTA_SECONDS) -> None:
        if not isfinite(fixed_delta_seconds) or fixed_delta_seconds <= 0.0:
            raise ValueError("fixed timestep must be finite and positive")
        self.fixed_delta_seconds = fixed_delta_seconds
        self._base: Any | None = None
        self._trial_index = 0

    def __enter__(self) -> SoloEvaluator:
        self._ensure_base()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Destroy the shared headless Panda application."""
        if self._base is None:
            return
        self._base.destroy()
        self._base = None

    def run_trial(
        self,
        *,
        controller_factory: ParameterizedControllerFactory,
        seed: int,
        duration_seconds: float = SOLO_TRIAL_DEFAULT_SECONDS,
    ) -> SoloTrialResult:
        """Evaluate one fresh controller using the trusted worker's tick order."""
        if not isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("trial duration must be finite and positive")

        base = self._ensure_base()
        self._trial_index += 1
        model = default_track_progress_model()
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = base.render.attachNewNode(f"solo-evaluator-{seed}-{self._trial_index}")
        try:
            add_racing_scene_collisions(physics_world=physics_world, render=root)
            spawn_pose = race_spawn_poses(
                1,
                model=model,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
                random_seed=seed,
                race_index=1,
            )[0]
            robot = create_robot_vehicle(
                world=physics_world,
                render=root,
                name=f"solo-evaluator-{seed}-{self._trial_index}-car",
                position=spawn_pose.position,
                heading_degrees=spawn_pose.heading_degrees,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            )
            physics_scene.vehicles.append(robot)
            runtime = RaceCarRuntime(
                robot=robot,
                tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose),
            )
            controller = controller_factory()
            elapsed_seconds = 0.0
            lap_crossing_times: list[float] = []
            previous_lap_count = 0

            while elapsed_seconds < duration_seconds:
                if not robot_is_eliminated(runtime.robot):
                    sensors, runtime.sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=runtime.robot,
                        track_model=model,
                        time_s=elapsed_seconds,
                        dt_s=self.fixed_delta_seconds,
                        previous_state=runtime.sensor_state,
                    )
                    command = _validated_controller_command(controller, sensors)
                    apply_robot_vehicle_command(robot=runtime.robot, command=command)

                physics_scene.step(self.fixed_delta_seconds)
                next_elapsed_seconds = min(duration_seconds, elapsed_seconds + self.fixed_delta_seconds)
                contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
                apply_wall_impact_damage(
                    physics_world=physics_world,
                    robots=(runtime.robot,),
                    fixed_time_step=physics_scene.fixed_time_step,
                )
                projection = project_track_position(model, robot_track_point(runtime.robot))
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=next_elapsed_seconds,
                    delta_seconds=self.fixed_delta_seconds,
                )
                while previous_lap_count < runtime.tracker.lap_count:
                    lap_crossing_times.append(next_elapsed_seconds)
                    previous_lap_count += 1
                elapsed_seconds = next_elapsed_seconds

            lap_durations = [
                crossing - (lap_crossing_times[index - 1] if index else 0.0)
                for index, crossing in enumerate(lap_crossing_times)
            ]
            damage = robot_score_damage(runtime.robot)
            return SoloTrialResult(
                seed=seed,
                elapsed_seconds=elapsed_seconds,
                raw_distance_m=runtime.tracker.best_distance_m,
                partial_laps=runtime.tracker.best_distance_m / model.total_length_m,
                lap_count=runtime.tracker.lap_count,
                damage=damage,
                survived=not robot_is_eliminated(runtime.robot) and damage < 1.0,
                wall_contact_seconds=runtime.tracker.wall_contact_seconds,
                max_speed_mps=runtime.max_speed_mps,
                first_lap_time_seconds=lap_crossing_times[0] if lap_crossing_times else None,
                best_lap_time_seconds=min(lap_durations) if lap_durations else None,
            )
        finally:
            root.removeNode()

    def _ensure_base(self) -> Any:
        if self._base is None:
            configure_headless_panda()
            showbase = cast(Any, import_module("direct.showbase.ShowBase"))
            self._base = showbase.ShowBase(windowType="none")
        return self._base


def controller_factory_from_module(
    module_reference: str | Path,
    *,
    function_name: str = "control",
) -> ParameterizedControllerFactory:
    """Return a factory that follows the public module/factory loading contract."""

    def create_controller() -> RobotController:
        return load_student_controller(module_reference, function_name=function_name)

    return create_controller


def run_solo_trial(
    *,
    controller_factory: ParameterizedControllerFactory,
    seed: int,
    duration_seconds: float = SOLO_TRIAL_DEFAULT_SECONDS,
    fixed_delta_seconds: float = SOLO_TRIAL_FIXED_DELTA_SECONDS,
) -> SoloTrialResult:
    """Run one trial with a short-lived evaluator."""
    with SoloEvaluator(fixed_delta_seconds=fixed_delta_seconds) as evaluator:
        return evaluator.run_trial(
            controller_factory=controller_factory,
            seed=seed,
            duration_seconds=duration_seconds,
        )


def _validated_controller_command(controller: RobotController, sensors: RobotSensors) -> RobotCommand:
    command = cast(object, controller(sensors))
    if not isinstance(command, RobotCommand):
        raise TypeError(f"controller must return racing.RobotCommand, got {type(command).__name__}")
    throttle = float(command.throttle)
    steer = float(command.steer)
    if not isfinite(throttle) or not isfinite(steer):
        raise ValueError("controller must return finite command values")
    return RobotCommand(throttle=throttle, steer=steer)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone evaluator command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="dotted controller module or Python file")
    parser.add_argument("--function", default="control", help="controller function name")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--seconds", type=float, default=SOLO_TRIAL_DEFAULT_SECONDS)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Evaluate one module and write a compact JSON result to stdout."""
    args = parse_args(arguments)
    result = run_solo_trial(
        controller_factory=controller_factory_from_module(str(args.module), function_name=str(args.function)),
        seed=int(args.seed),
        duration_seconds=float(args.seconds),
    )
    print(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
