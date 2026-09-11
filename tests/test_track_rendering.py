from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import pytest

from racing.graphics.track_mesh import clean_offset_path
from racing.graphics.track_rendering import (
    START_FINISH_ARGYLE_FLOOR_LENGTH,
    START_FINISH_ARGYLE_FLOOR_WIDTH,
    START_FINISH_ARGYLE_TEXTURE_ASPECT_RATIO,
    START_FINISH_BANNER_HEIGHT,
    START_FINISH_BANNER_OVERHANG,
    START_FINISH_BANNER_POLE_THICKNESS,
    START_FINISH_BANNER_POLE_WALL_CLEARANCE,
    START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO,
    START_FINISH_FORMULA_VERTICAL_MARGIN_FRACTION,
    TRACK_EDGE_BUFFER,
    TRACK_KERB_INNER_DISTANCE,
    TRACK_KERB_OUTER_DISTANCE,
    TRACK_SURFACE_Y,
    TRACK_WALL_THICKNESS,
    add_racing_scene_collisions,
    start_finish_banner_side_distances,
    start_finish_render_pose,
    start_finish_track_slice,
)
from racing.physics import create_physics_world
from racing.race.progress import default_track_progress_model
from racing.race.runtime import seeded_race_start_finish_pose
from racing.track.world import (
    TRACK_ID_BLUE_RIDGE_LOOP,
    TRACK_ID_DOGWOOD_LOOP,
    TRACK_ID_HARBOR_LOOP,
    TRACK_ID_PINE_SWITCHBACKS,
    TRACK_ID_STADIUM_LOOP,
    TRACK_WIDTH,
    sampled_track_centerline,
    track_layout_by_id,
)

GENERATED_TRACK_IDS = (
    TRACK_ID_STADIUM_LOOP,
    TRACK_ID_HARBOR_LOOP,
    TRACK_ID_DOGWOOD_LOOP,
    TRACK_ID_BLUE_RIDGE_LOOP,
    TRACK_ID_PINE_SWITCHBACKS,
)


def test_kerb_center_rays_hit_flat_floor_collider() -> None:
    core = cast(Any, import_module("panda3d.core"))
    world = create_physics_world()
    render = core.NodePath("render")
    samples = sampled_track_centerline(samples_per_segment=10)
    kerb_center_distance = (TRACK_KERB_INNER_DISTANCE + TRACK_KERB_OUTER_DISTANCE) / 2.0

    add_racing_scene_collisions(physics_world=world, render=render, samples=samples)

    for side in (-1, 1):
        kerb_center_points = clean_offset_path(samples, side * kerb_center_distance, 0.0)
        for point in kerb_center_points[::25]:
            hit = world.rayTestClosest(
                core.Point3(point[0], 3.0, point[2]),
                core.Point3(point[0], -1.0, point[2]),
            )

            assert hit.hasHit()
            assert hit.getNode().getName() == "grass-and-track-floor"
            assert float(hit.getHitPos()[1]) == pytest.approx(TRACK_SURFACE_Y, abs=1e-3)
            assert float(hit.getHitNormal()[1]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("track_id", GENERATED_TRACK_IDS)
def test_generated_track_wall_offsets_remain_well_formed(track_id: str) -> None:
    layout = track_layout_by_id(track_id)
    samples = sampled_track_centerline(layout.points, samples_per_segment=10)
    wall_outside_distance = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS

    for side in (-1, 1):
        cleaned_wall = clean_offset_path(samples, side * wall_outside_distance, 0.0)

        assert len(cleaned_wall) >= len(samples) * 0.85


def test_problem_seed_start_finish_poles_clear_outside_wall() -> None:
    model = default_track_progress_model()
    samples = sampled_track_centerline(samples_per_segment=10)
    pole_radius = START_FINISH_BANNER_POLE_THICKNESS / 2
    required_clearance = pole_radius + START_FINISH_BANNER_POLE_WALL_CLEARANCE

    for seed in (2, 5, 6):
        pose = seeded_race_start_finish_pose(model=model, random_seed=seed, race_index=1)
        render_pose = start_finish_render_pose(samples=samples, position=pose.position)
        track_slice = start_finish_track_slice(samples=samples, position=pose.position)
        side_distances = start_finish_banner_side_distances(
            samples=samples,
            position=pose.position,
        )

        assert render_pose.position.x == pytest.approx(track_slice.pose.position.x)
        assert render_pose.position.z == pytest.approx(track_slice.pose.position.z)
        assert render_pose.heading_degrees == pytest.approx(track_slice.pose.heading_degrees)

        for pole_distance, wall_distance in zip(
            side_distances,
            (track_slice.negative_wall_distance, track_slice.positive_wall_distance),
            strict=True,
        ):
            assert pole_distance >= wall_distance + required_clearance - 1e-9


def test_start_finish_banner_is_tall_enough_for_formula_logo() -> None:
    wall_outside_distance = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS
    nominal_side_distance = (
        wall_outside_distance + START_FINISH_BANNER_POLE_THICKNESS / 2 + START_FINISH_BANNER_POLE_WALL_CLEARANCE
    )
    nominal_width = nominal_side_distance * 2 + START_FINISH_BANNER_OVERHANG
    available_height = START_FINISH_BANNER_HEIGHT * (1 - 2 * START_FINISH_FORMULA_VERTICAL_MARGIN_FRACTION)
    full_width_logo_height = nominal_width / START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO

    assert available_height >= full_width_logo_height


def test_start_finish_floor_argyle_matches_texture_aspect_ratio() -> None:
    assert (
        pytest.approx(START_FINISH_ARGYLE_TEXTURE_ASPECT_RATIO)
        == START_FINISH_ARGYLE_FLOOR_WIDTH / START_FINISH_ARGYLE_FLOOR_LENGTH
    )
