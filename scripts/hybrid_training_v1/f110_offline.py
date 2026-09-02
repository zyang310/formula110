"""Shared local simulation helpers for Formula 110 offline tooling."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from racing.graphics.panda_config import configure_headless_panda  # noqa: E402
from racing.graphics.track_rendering import TRACK_SURFACE_Y, add_racing_scene_collisions  # noqa: E402
from racing.physics import (  # noqa: E402
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
    vehicle_spawn_height,
)
from racing.race.progress import (  # noqa: E402
    default_track_progress_model,
    project_track_position,
    track_pose_at_distance,
)
from racing.race.runtime import (  # noqa: E402
    RaceCarRuntime,
    RaceSpawnPose,
    lap_progress_tracker_for_spawn_pose,
    race_contact_states,
    race_spawn_poses,
    robot_is_eliminated,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import build_robot_sensors  # noqa: E402
from racing.student.api import RobotCommand, RobotController, RobotSensors, load_student_controller  # noqa: E402
from racing.track.spatial import track_forward_vector, track_left_vector  # noqa: E402

CONTROL_HZ = 60.0
FIXED_DELTA_SECONDS = 1.0 / CONTROL_HZ


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """Configuration for one local single-car trial."""

    seed: int = 110
    seconds: float = 30.0
    race_index: int = 1
    start_progress_m: float | None = None
    start_speed_mps: float = 0.0
    start_lateral_offset_m: float = 0.0
    start_heading_error_degrees: float = 0.0
    record_trace: bool = False


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Serializable metrics mirroring the Gradescope single-car trial."""

    ok: bool
    seed: int
    elapsed_seconds: float
    raw_distance_m: float
    partial_laps: float
    lap_count: int
    survived: bool
    final_damage: float
    wall_contact_seconds: float
    max_speed_mps: float
    final_speed_mps: float
    first_lap_time_seconds: float | None
    best_lap_time_seconds: float | None
    start_progress_m: float
    trace: tuple[dict[str, float], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ControllerFactory = Callable[[], RobotController]


def controller_factory_from_module(
    module_reference: str | Path,
    *,
    function_name: str = "control",
) -> ControllerFactory:
    """Return a factory that loads a fresh student controller instance per trial."""

    def factory() -> RobotController:
        return load_student_controller(module_reference, function_name=function_name)

    return factory


class HeadlessRaceSession:
    """Reusable headless Panda/Bullet host for short local race trials."""

    def __init__(self) -> None:
        configure_headless_panda()
        showbase = cast(Any, import_module("direct.showbase.ShowBase"))
        self._base = showbase.ShowBase(windowType="none")
        self._trial_index = 0

    def close(self) -> None:
        self._base.destroy()

    def __enter__(self) -> HeadlessRaceSession:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def run_trial(self, controller_factory: ControllerFactory, config: TrialConfig) -> TrialResult:
        if config.seconds <= 0.0:
            raise ValueError("trial duration must be positive")
        self._trial_index += 1
        model = default_track_progress_model()
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = self._base.render.attachNewNode(f"offline-trial-{config.seed}-{self._trial_index}")
        try:
            add_racing_scene_collisions(physics_world=physics_world, render=root)
            spawn_pose = _spawn_pose(model=model, config=config)
            robot = create_robot_vehicle(
                world=physics_world,
                render=root,
                name=f"offline-car-{config.seed}-{self._trial_index}",
                position=spawn_pose.position,
                heading_degrees=spawn_pose.heading_degrees,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            )
            _set_initial_forward_speed(
                robot,
                heading_degrees=spawn_pose.heading_degrees,
                speed_mps=config.start_speed_mps,
            )
            physics_scene.vehicles.append(robot)
            runtime = RaceCarRuntime(
                robot=robot,
                tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose),
            )
            controller = controller_factory()
            elapsed_seconds = 0.0
            previous_lap_count = 0
            lap_crossing_times: list[float] = []
            trace: list[dict[str, float]] = []
            while elapsed_seconds < config.seconds:
                command = RobotCommand()
                sensors: RobotSensors | None = None
                if not robot_is_eliminated(runtime.robot):
                    sensors, runtime.sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=runtime.robot,
                        track_model=model,
                        time_s=elapsed_seconds,
                        dt_s=FIXED_DELTA_SECONDS,
                        previous_state=runtime.sensor_state,
                    )
                    command = controller(sensors)
                    apply_robot_vehicle_command(robot=runtime.robot, command=command)

                physics_scene.step(FIXED_DELTA_SECONDS)
                next_elapsed_seconds = min(config.seconds, elapsed_seconds + FIXED_DELTA_SECONDS)
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
                    delta_seconds=FIXED_DELTA_SECONDS,
                )
                if config.record_trace:
                    speed_mps = abs(float(runtime.robot.vehicle.getCurrentSpeedKmHour()) / 3.6)
                    trace.append(
                        {
                            "time_s": next_elapsed_seconds,
                            "progress_m": projection.progress_distance_m,
                            "raw_distance_m": runtime.tracker.best_distance_m,
                            "speed_mps": speed_mps,
                            "damage": robot_score_damage(runtime.robot),
                            "wall_contact": contact_state.wall_contact,
                            "throttle": command.throttle,
                            "steer": command.steer,
                            "heading_error_degrees": 0.0 if sensors is None else sensors.camera.heading_error_degrees,
                            "center_offset_m": 0.0 if sensors is None else sensors.camera.center_offset_m,
                        }
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
            final_speed_mps = abs(float(robot.vehicle.getCurrentSpeedKmHour()) / 3.6)
            return TrialResult(
                ok=True,
                seed=config.seed,
                elapsed_seconds=elapsed_seconds,
                raw_distance_m=runtime.tracker.best_distance_m,
                partial_laps=runtime.tracker.best_distance_m / model.total_length_m,
                lap_count=runtime.tracker.lap_count,
                survived=not robot_is_eliminated(runtime.robot) and damage < 1.0,
                final_damage=damage,
                wall_contact_seconds=runtime.tracker.wall_contact_seconds,
                max_speed_mps=runtime.max_speed_mps,
                final_speed_mps=final_speed_mps,
                first_lap_time_seconds=lap_crossing_times[0] if lap_crossing_times else None,
                best_lap_time_seconds=min(lap_durations) if lap_durations else None,
                start_progress_m=spawn_pose.progress_distance_m,
                trace=tuple(trace),
            )
        except BaseException as error:
            return TrialResult(
                ok=False,
                seed=config.seed,
                elapsed_seconds=0.0,
                raw_distance_m=0.0,
                partial_laps=0.0,
                lap_count=0,
                survived=False,
                final_damage=1.0,
                wall_contact_seconds=0.0,
                max_speed_mps=0.0,
                final_speed_mps=0.0,
                first_lap_time_seconds=None,
                best_lap_time_seconds=None,
                start_progress_m=config.start_progress_m or 0.0,
                error=f"{type(error).__name__}: {error}"[:1000],
            )
        finally:
            with contextlib.suppress(Exception):
                root.removeNode()


def run_trials(
    controller_factory: ControllerFactory,
    configs: tuple[TrialConfig, ...],
) -> tuple[TrialResult, ...]:
    """Run a batch of single-car local trials in one headless session."""
    with HeadlessRaceSession() as session:
        return tuple(session.run_trial(controller_factory, config) for config in configs)


def _spawn_pose(*, model: Any, config: TrialConfig) -> RaceSpawnPose:
    if config.start_progress_m is None:
        spawn_pose = race_spawn_poses(
            1,
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=config.seed,
            race_index=config.race_index,
        )[0]
        if config.start_lateral_offset_m == 0.0 and config.start_heading_error_degrees == 0.0:
            return spawn_pose
        track_pose = track_pose_at_distance(model, spawn_pose.progress_distance_m)
    else:
        track_pose = track_pose_at_distance(model, config.start_progress_m)
    left_x, left_z = track_left_vector(track_pose.heading_degrees)
    heading_degrees = track_pose.heading_degrees + config.start_heading_error_degrees
    return RaceSpawnPose(
        position=(
            track_pose.position.x + left_x * config.start_lateral_offset_m,
            vehicle_spawn_height(FORMULA_VEHICLE_PHYSICS_CONFIG, surface_y=TRACK_SURFACE_Y),
            track_pose.position.z + left_z * config.start_lateral_offset_m,
        ),
        heading_degrees=heading_degrees,
        progress_distance_m=track_pose.progress_distance_m,
    )


def _set_initial_forward_speed(robot: Any, *, heading_degrees: float, speed_mps: float) -> None:
    if speed_mps == 0.0:
        return
    core = cast(Any, import_module("panda3d.core"))
    forward_x, forward_z = track_forward_vector(heading_degrees)
    body = robot.chassis_np.node()
    if hasattr(body, "setLinearVelocity"):
        body.setLinearVelocity(core.Vec3(forward_x * speed_mps, 0.0, forward_z * speed_mps))
    if hasattr(body, "setActive"):
        body.setActive(True)


class ScriptedManeuverController:
    """Play a short open-loop maneuver before falling back to a legal controller."""

    def __init__(
        self,
        segments: tuple[tuple[int, float, float], ...],
        fallback: RobotController,
    ) -> None:
        self._segments = segments
        self._fallback = fallback

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        remaining_tick = sensors.tick
        for duration_ticks, throttle, steer in self._segments:
            if remaining_tick < duration_ticks:
                return RobotCommand(throttle=throttle, steer=steer)
            remaining_tick -= duration_ticks
        return self._fallback(sensors)
