from __future__ import annotations

import pytest

from racing.race.progress import track_progress_model_for_layout
from racing.track.world import (
    CAROLINA_LOOP_LAYOUT,
    CAROLINA_LOOP_START_POSITION,
    MUGELLO_SHORT_LAYOUT,
    TRACK_ID_BLUE_RIDGE_LOOP,
    TRACK_ID_CAROLINA_LOOP,
    TRACK_ID_DOGWOOD_LOOP,
    TRACK_ID_HARBOR_LOOP,
    TRACK_ID_MUGELLO_SHORT,
    TRACK_ID_PINE_SWITCHBACKS,
    TRACK_ID_STADIUM_LOOP,
    TrackPoint,
    sampled_track_centerline,
    total_track_length,
    track_bounds,
    track_layout_by_id,
    track_layout_ids,
)

GENERATED_TRACK_IDS = (
    TRACK_ID_STADIUM_LOOP,
    TRACK_ID_HARBOR_LOOP,
    TRACK_ID_DOGWOOD_LOOP,
    TRACK_ID_BLUE_RIDGE_LOOP,
    TRACK_ID_PINE_SWITCHBACKS,
)


def test_default_track_layout_is_available() -> None:
    assert TRACK_ID_MUGELLO_SHORT in track_layout_ids()
    assert track_layout_by_id(TRACK_ID_MUGELLO_SHORT).track_id == TRACK_ID_MUGELLO_SHORT


def test_carolina_loop_is_a_distinct_available_track() -> None:
    layout = track_layout_by_id(TRACK_ID_CAROLINA_LOOP)

    assert TRACK_ID_CAROLINA_LOOP in track_layout_ids()
    assert layout.points == CAROLINA_LOOP_LAYOUT
    assert layout.start_position == CAROLINA_LOOP_START_POSITION
    assert total_track_length(layout.points) > total_track_length(MUGELLO_SHORT_LAYOUT)
    assert track_bounds(layout.points, margin=0.0) != track_bounds(MUGELLO_SHORT_LAYOUT, margin=0.0)


def test_generated_tracks_increase_in_declared_and_geometric_complexity() -> None:
    layouts = tuple(track_layout_by_id(track_id) for track_id in GENERATED_TRACK_IDS)

    assert tuple(layout.complexity for layout in layouts) == (
        "beginner",
        "easy",
        "intermediate",
        "advanced",
        "expert",
    )
    assert tuple(len(layout.points) for layout in layouts) == (16, 18, 22, 26, 30)
    assert tuple(total_track_length(layout.points) for layout in layouts) == tuple(
        sorted(total_track_length(layout.points) for layout in layouts)
    )
    assert len({layout.points for layout in layouts}) == len(layouts)

    opposite_direction_turn_counts = tuple(
        sum(
            _orientation(layout.points[index - 1], point, layout.points[(index + 1) % len(layout.points)]) < 0.0
            for index, point in enumerate(layout.points)
        )
        for layout in layouts
    )
    assert opposite_direction_turn_counts == tuple(sorted(opposite_direction_turn_counts))
    assert opposite_direction_turn_counts[0] >= 2


@pytest.mark.parametrize("track_id", GENERATED_TRACK_IDS)
def test_generated_track_progress_models_start_cleanly(track_id: str) -> None:
    layout = track_layout_by_id(track_id)
    model = track_progress_model_for_layout(track_id)

    assert model.points[0].x == pytest.approx(layout.start_position.x)
    assert model.points[0].z == pytest.approx(layout.start_position.z)
    assert all(segment_length > 0.0 for segment_length in model.segment_lengths)


@pytest.mark.parametrize("track_id", GENERATED_TRACK_IDS)
def test_generated_track_centerlines_do_not_self_intersect(track_id: str) -> None:
    layout = track_layout_by_id(track_id)
    samples = sampled_track_centerline(layout.points, samples_per_segment=10)
    signed_sample_turns = tuple(
        _orientation(samples[index - 1], point, samples[(index + 1) % len(samples)])
        for index, point in enumerate(samples)
    )

    assert not _self_intersecting_segment_pairs(samples)
    assert min(signed_sample_turns) < -0.05
    assert max(signed_sample_turns) > 0.05


def test_unknown_track_layout_reports_valid_ids() -> None:
    with pytest.raises(ValueError, match=TRACK_ID_MUGELLO_SHORT):
        track_layout_by_id("missing")


def test_track_length_and_bounds_are_positive() -> None:
    bounds = track_bounds(MUGELLO_SHORT_LAYOUT)

    assert total_track_length(MUGELLO_SHORT_LAYOUT) > 0.0
    assert bounds.width > 0.0
    assert bounds.length > 0.0


def test_sampled_track_centerline_requires_positive_samples() -> None:
    with pytest.raises(ValueError, match="samples_per_segment"):
        sampled_track_centerline(MUGELLO_SHORT_LAYOUT, samples_per_segment=0)


def _self_intersecting_segment_pairs(points: tuple[TrackPoint, ...]) -> tuple[tuple[int, int], ...]:
    intersections: list[tuple[int, int]] = []
    for first_index, first_start in enumerate(points):
        first_end = points[(first_index + 1) % len(points)]
        for second_index in range(first_index + 1, len(points)):
            if min(second_index - first_index, len(points) - (second_index - first_index)) <= 1:
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % len(points)]
            if _segments_properly_intersect(first_start, first_end, second_start, second_end):
                intersections.append((first_index, second_index))
    return tuple(intersections)


def _segments_properly_intersect(
    first_start: TrackPoint,
    first_end: TrackPoint,
    second_start: TrackPoint,
    second_end: TrackPoint,
) -> bool:
    first_side_start = _orientation(first_start, first_end, second_start)
    first_side_end = _orientation(first_start, first_end, second_end)
    second_side_start = _orientation(second_start, second_end, first_start)
    second_side_end = _orientation(second_start, second_end, first_end)
    return first_side_start * first_side_end < 0.0 and second_side_start * second_side_end < 0.0


def _orientation(start: TrackPoint, end: TrackPoint, point: TrackPoint) -> float:
    return (end.x - start.x) * (point.z - start.z) - (end.z - start.z) * (point.x - start.x)
