"""Capture world X/Z trajectories for a few controllers (and the human replay) on seed 110."""

import json
import sys
from pathlib import Path

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
from racing.student.api import RobotCommand, load_student_controller
from scripts.controller_training.evaluator import SoloEvaluator

DT = 1.0 / 60.0
OUT = Path(sys.argv[1])
HUMAN = Path("artifacts/human-driving.jsonl")


class Replay:
    def __init__(self, commands):
        self.commands = commands
        self.index = 0
        self.observed = []

    def __call__(self, sensors):
        self.observed.append((sensors.imu.heading_degrees, sensors.odometry.speed_mps))
        throttle, steer = self.commands[self.index] if self.index < len(self.commands) else (0.0, 0.0)
        self.index += 1
        return RobotCommand(throttle=throttle, steer=steer)


def run(evaluator, controller, seed, seconds, tag):
    base = evaluator._ensure_base()
    model = default_track_progress_model()
    world = create_physics_world()
    scene = PhysicsScene(world=world, vehicles=[])
    root = base.render.attachNewNode(f"cap-{tag}")
    add_racing_scene_collisions(physics_world=world, render=root)
    pose = race_spawn_poses(1, model=model, config=FORMULA_VEHICLE_PHYSICS_CONFIG, random_seed=seed, race_index=1)[0]
    robot = create_robot_vehicle(
        world=world,
        render=root,
        name=f"cap-{tag}-car",
        position=pose.position,
        heading_degrees=pose.heading_degrees,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )
    scene.vehicles.append(robot)
    runtime = RaceCarRuntime(robot=robot, tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=pose))
    elapsed = 0.0
    rows = []
    crossings = []
    previous_laps = 0
    while elapsed < seconds - 1e-9:
        command = RobotCommand(throttle=0.0, steer=0.0)
        speed = 0.0
        if not robot_is_eliminated(runtime.robot):
            sensors, runtime.sensor_state = build_robot_sensors(
                physics_world=world,
                robot=runtime.robot,
                track_model=model,
                time_s=elapsed,
                dt_s=DT,
                previous_state=runtime.sensor_state,
            )
            command = controller(sensors)
            speed = sensors.odometry.speed_mps
            apply_robot_vehicle_command(robot=runtime.robot, command=command)
        point = robot_track_point(runtime.robot)
        rows.append(
            [
                round(point.x, 3),
                round(point.z, 3),
                round(speed, 2),
                round(float(command.steer), 3),
                round(float(command.throttle), 3),
                round(runtime.tracker.wall_contact_seconds, 3),
            ]
        )
        scene.step(DT)
        nxt = min(seconds, elapsed + DT)
        contact = race_contact_states(physics_world=world, runtimes=(runtime,))[0]
        apply_wall_impact_damage(physics_world=world, robots=(runtime.robot,), fixed_time_step=scene.fixed_time_step)
        projection = project_track_position(model, robot_track_point(runtime.robot))
        update_race_runtime_after_step(
            runtime=runtime, projection=projection, contact_state=contact, elapsed_seconds=nxt, delta_seconds=DT
        )
        while previous_laps < runtime.tracker.lap_count:
            crossings.append(round(nxt, 4))
            previous_laps += 1
        elapsed = nxt
    root.removeNode()
    laps = [c - (crossings[i - 1] if i else 0.0) for i, c in enumerate(crossings)]
    return {
        "rows": rows,
        "crossings": crossings,
        "best_lap": round(min(laps), 4) if laps else None,
        "first_lap": crossings[0] if crossings else None,
        "distance_m": round(runtime.tracker.best_distance_m, 2),
        "damage": round(robot_score_damage(runtime.robot), 4),
        "wall_contact_s": round(runtime.tracker.wall_contact_seconds, 3),
        "spawn_heading": pose.heading_degrees,
    }


def main():
    human_rows = [json.loads(line) for line in HUMAN.read_text().splitlines() if line.strip()]
    commands = [(r["command"]["throttle"], r["command"]["steer"]) for r in human_rows]
    out = {}
    with SoloEvaluator() as evaluator:
        for tag, module in (("minimum", "controllers.minimum_viable"), ("v27", "controllers.race_faster")):
            out[tag] = run(evaluator, load_student_controller(module), 110, 30.0, tag)
            print(tag, {k: v for k, v in out[tag].items() if k != "rows"})
        replay = Replay(commands)
        out["human"] = run(evaluator, replay, 110, len(commands) * DT, "human")
        print("human", {k: v for k, v in out["human"].items() if k != "rows"})
        for k in (0, 60, 120, 240, 360, 480, 600):
            if k < len(replay.observed):
                rec = human_rows[k]["sensors"]
                print(
                    "replay check tick",
                    k,
                    "sim",
                    [round(x, 3) for x in replay.observed[k]],
                    "rec",
                    [round(rec["imu"]["heading_degrees"], 3), round(rec["odometry"]["speed_mps"], 3)],
                )
    OUT.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
