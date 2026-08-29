from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import pytest

from racing.physics import (
    DEFAULT_VEHICLE_PHYSICS_CONFIG,
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    VehiclePhysicsConfig,
    apply_robot_vehicle_command,
    attach_static_box,
    create_physics_world,
    create_robot_vehicle,
    resolve_vehicle_actuator_command,
    vehicle_collision_bounds,
    vehicle_spawn_height,
    wall_damage_from_impact_impulse,
    wall_damage_reference_impulse_n_s,
    wheel_axis_points,
    wheel_connection_points,
)
from racing.student.api import RobotCommand


def test_default_physics_config_is_formula_car() -> None:
    assert DEFAULT_VEHICLE_PHYSICS_CONFIG is FORMULA_VEHICLE_PHYSICS_CONFIG


def test_formula_collision_bounds_use_convex_hull() -> None:
    bounds = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG)

    assert bounds.half_width == pytest.approx(0.630)
    assert bounds.half_length == pytest.approx(1.150)
    assert bounds.height == pytest.approx(0.660)


def test_wheel_points_are_symmetric() -> None:
    config = FORMULA_VEHICLE_PHYSICS_CONFIG

    assert wheel_axis_points(config) == (
        (-config.wheel_track_half_width, 0.0, config.wheelbase_half_length),
        (config.wheel_track_half_width, 0.0, config.wheelbase_half_length),
        (-config.wheel_track_half_width, 0.0, -config.wheelbase_half_length),
        (config.wheel_track_half_width, 0.0, -config.wheelbase_half_length),
    )
    assert all(point[1] == config.wheel_connection_height for point in wheel_connection_points(config))


def test_resolve_vehicle_command_maps_forward_throttle_and_steering() -> None:
    command = resolve_vehicle_actuator_command(
        command=RobotCommand(throttle=0.5, steer=0.25),
        current_speed_kmh=0.0,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )

    assert command.steering_degrees == pytest.approx(FORMULA_VEHICLE_PHYSICS_CONFIG.max_steering_degrees * 0.25)
    assert command.engine_force == pytest.approx(FORMULA_VEHICLE_PHYSICS_CONFIG.max_engine_force * 0.5)
    assert command.brake_force == 0.0


def test_forward_motion_rotates_top_of_wheel_toward_front() -> None:
    core = cast(Any, import_module("panda3d.core"))
    config = FORMULA_VEHICLE_PHYSICS_CONFIG
    world = create_physics_world()
    render = core.NodePath("render")
    attach_static_box(
        world=world,
        render=render,
        name="wheel-spin-test-floor",
        position=(0.0, -0.05, 0.0),
        half_extents=(20.0, 0.05, 20.0),
        friction=1.0,
    )
    robot = create_robot_vehicle(
        world=world,
        render=render,
        name="wheel-spin-test-car",
        position=(0.0, vehicle_spawn_height(config), 0.0),
        config=config,
    )
    scene = PhysicsScene(world=world, vehicles=[robot])
    apply_robot_vehicle_command(robot=robot, command=RobotCommand(throttle=0.2))

    for _ in range(12):
        scene.step(1 / 120)

    front_right_wheel_top = robot.wheel_nodes[1].getQuat(render).xform(core.Vec3(0.0, 0.0, 1.0))
    assert float(robot.chassis_np.getZ(render)) > 0.0
    assert float(front_right_wheel_top[2]) > 0.0


def test_resolve_vehicle_command_brakes_before_reverse() -> None:
    command = resolve_vehicle_actuator_command(
        command=RobotCommand(throttle=-0.7),
        current_speed_kmh=12.0,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )

    assert command.engine_force == 0.0
    assert command.brake_force > 0.0
    assert command.next_pending_drive_direction == -1


def test_pending_direction_latch_blocks_positive_throttle_until_nearly_stopped() -> None:
    command = resolve_vehicle_actuator_command(
        command=RobotCommand(throttle=0.9),
        current_speed_kmh=61.0,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        pending_drive_direction=-1,
    )

    assert command.engine_force == 0.0
    assert command.brake_force > 0.0
    assert command.next_pending_drive_direction == 1


def test_zero_throttle_clears_the_pending_direction_latch() -> None:
    command = resolve_vehicle_actuator_command(
        command=RobotCommand(throttle=0.0),
        current_speed_kmh=61.0,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        pending_drive_direction=-1,
    )

    assert command.engine_force == 0.0
    assert command.brake_force == 0.0
    assert command.next_pending_drive_direction == 0


def test_vehicle_spawn_height_requires_positive_wheel_radius() -> None:
    with pytest.raises(ValueError, match="wheel_radius"):
        vehicle_spawn_height(VehiclePhysicsConfig(wheel_radius=0.0))


def test_wall_damage_scales_to_full_damage_at_reference_impulse() -> None:
    reference = wall_damage_reference_impulse_n_s(FORMULA_VEHICLE_PHYSICS_CONFIG)

    assert wall_damage_from_impact_impulse(reference, FORMULA_VEHICLE_PHYSICS_CONFIG) == 1.0
    assert wall_damage_from_impact_impulse(reference / 2, FORMULA_VEHICLE_PHYSICS_CONFIG) == pytest.approx(0.25)


def test_static_box_heading_rotates_around_physics_up_axis() -> None:
    core = cast(Any, import_module("panda3d.core"))
    world = create_physics_world()
    render = core.NodePath("render")

    attach_static_box(
        world=world,
        render=render,
        name="rotated-flat-box",
        position=(0.0, 0.0, 0.0),
        half_extents=(2.0, 0.1, 0.25),
        heading_degrees=37.0,
    )

    hit = world.rayTestClosest(core.Point3(0.0, 1.0, 0.0), core.Point3(0.0, -1.0, 0.0))

    assert hit.hasHit()
    normal = hit.getHitNormal()
    assert float(normal[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(normal[1]) == pytest.approx(1.0, abs=1e-5)
    assert float(normal[2]) == pytest.approx(0.0, abs=1e-5)
