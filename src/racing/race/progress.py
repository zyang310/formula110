"""Pure lap-progress helpers for student racing."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from math import atan2, degrees, hypot

from racing.track.world import (
    START_POSITION,
    TRACK_ID_MUGELLO_SHORT,
    TrackPoint,
    sampled_track_centerline,
    track_layout_by_id,
)

TrackProgressSegmentProjection = tuple[float, float, float, float, float, float, float, float]


def _empty_lap_times() -> list[float]:
    return []


@dataclass(frozen=True, slots=True)
class TrackProjection:
    """Projection of a world position onto the track centerline."""

    position: TrackPoint
    nearest_center: TrackPoint
    progress_distance_m: float
    lap_progress: float
    signed_distance_to_center_m: float
    heading_degrees: float


@dataclass(frozen=True, slots=True)
class TrackPose:
    """Position and heading sampled at a distance along the track."""

    position: TrackPoint
    heading_degrees: float
    progress_distance_m: float


@dataclass(frozen=True, slots=True)
class TrackProgressModel:
    """Precomputed centerline distances used to measure lap progress."""

    points: tuple[TrackPoint, ...]
    segment_lengths: tuple[float, ...]
    cumulative_lengths: tuple[float, ...]
    total_length_m: float
    segment_projections: tuple[TrackProgressSegmentProjection, ...] = ()
    segment_headings_degrees: tuple[float, ...] = ()


@dataclass(slots=True)
class LapProgressTracker:
    """Mutable lap, distance, and contact accounting for one car."""

    total_length_m: float
    starting_progress_distance_m: float | None = None
    last_progress_distance_m: float | None = None
    last_elapsed_seconds: float | None = None
    unwrapped_progress_distance_m: float = 0.0
    best_distance_m: float = 0.0
    last_progress_delta_m: float = 0.0
    last_counted_progress_delta_m: float = 0.0
    penalized_distance_m: float = 0.0
    wall_contact_seconds: float = 0.0
    car_contact_seconds: float = 0.0
    wall_contact_streak_seconds: float = 0.0
    car_contact_streak_seconds: float = 0.0
    max_wall_contact_streak_seconds: float = 0.0
    max_car_contact_streak_seconds: float = 0.0
    lap_time_seconds: float | None = None
    lap_times_seconds: list[float] = field(default_factory=_empty_lap_times)

    @property
    def completed_lap(self) -> bool:
        """Return whether this tracker has completed at least one lap."""
        return self.lap_count > 0

    @property
    def lap_count(self) -> int:
        """Return the number of completed laps."""
        return len(self.lap_times_seconds)

    def update(
        self,
        progress_distance_m: float,
        elapsed_seconds: float,
        *,
        distance_counts: bool = True,
        wall_contact: bool = False,
        car_contact: bool = False,
    ) -> None:
        """Update lap progress and contact timers from one simulation tick."""
        if self.total_length_m <= 0:
            raise ValueError("total_length_m must be positive")

        wrapped_progress = progress_distance_m % self.total_length_m
        if self.last_progress_distance_m is None:
            if self.starting_progress_distance_m is None:
                self.starting_progress_distance_m = wrapped_progress
                self.last_progress_distance_m = wrapped_progress
                self.last_elapsed_seconds = elapsed_seconds
                return
            self.last_progress_distance_m = self.starting_progress_distance_m % self.total_length_m
            if self.last_elapsed_seconds is None:
                self.last_elapsed_seconds = 0.0

        elapsed_delta = (
            0.0 if self.last_elapsed_seconds is None else max(0.0, elapsed_seconds - self.last_elapsed_seconds)
        )
        delta = wrapped_progress - self.last_progress_distance_m
        if delta < -self.total_length_m / 2:
            delta += self.total_length_m
        elif delta > self.total_length_m / 2:
            delta -= self.total_length_m

        self.last_progress_delta_m = delta
        if distance_counts:
            self.unwrapped_progress_distance_m += delta
            self.last_counted_progress_delta_m = delta
        else:
            self.penalized_distance_m += max(0.0, delta)
            self.last_counted_progress_delta_m = 0.0
        if wall_contact:
            self.wall_contact_seconds += elapsed_delta
            self.wall_contact_streak_seconds += elapsed_delta
            self.max_wall_contact_streak_seconds = max(
                self.max_wall_contact_streak_seconds, self.wall_contact_streak_seconds
            )
        else:
            self.wall_contact_streak_seconds = 0.0
        if car_contact:
            self.car_contact_seconds += elapsed_delta
            self.car_contact_streak_seconds += elapsed_delta
            self.max_car_contact_streak_seconds = max(
                self.max_car_contact_streak_seconds, self.car_contact_streak_seconds
            )
        else:
            self.car_contact_streak_seconds = 0.0
        self.last_progress_distance_m = wrapped_progress
        self.last_elapsed_seconds = elapsed_seconds
        self.best_distance_m = max(self.best_distance_m, self.unwrapped_progress_distance_m)
        completed_laps = int(self.best_distance_m // self.total_length_m)
        while len(self.lap_times_seconds) < completed_laps:
            self.lap_times_seconds.append(elapsed_seconds)
        if self.lap_time_seconds is None and self.lap_times_seconds:
            self.lap_time_seconds = self.lap_times_seconds[0]


def default_track_progress_model() -> TrackProgressModel:
    """Use the default track layout for lap-progress calculations."""
    return track_progress_model_for_layout(TRACK_ID_MUGELLO_SHORT)


@lru_cache(maxsize=16)
def track_progress_model_for_layout(track_id: str) -> TrackProgressModel:
    """Build lap-progress lookup data for one named track layout."""
    layout = track_layout_by_id(track_id)
    oriented_points = sampled_track_centerline(layout.points, samples_per_segment=10)
    start_line = TrackPoint(layout.start_position.x, layout.start_position.z, START_POSITION.label)
    return build_track_progress_model(_rebased_track_points(oriented_points, start_line=start_line))


def build_track_progress_model(points: tuple[TrackPoint, ...]) -> TrackProgressModel:
    """Precompute segment lengths and headings for a closed track."""
    if len(points) < 3:
        raise ValueError("a track progress model needs at least three points")

    segment_lengths: list[float] = []
    segment_projections: list[TrackProgressSegmentProjection] = []
    segment_headings_degrees: list[float] = []
    cumulative_lengths = [0.0]
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx = end.x - start.x
        dz = end.z - start.z
        length = hypot(dx, dz)
        if length <= 0:
            raise ValueError("track progress model cannot contain zero-length segments")
        segment_projections.append(
            (start.x, start.z, dx, dz, 1.0 / (length * length), -dz / length, dx / length, degrees(atan2(dx, dz)))
        )
        segment_headings_degrees.append(degrees(atan2(dx, dz)))
        segment_lengths.append(length)
        cumulative_lengths.append(cumulative_lengths[-1] + length)

    return TrackProgressModel(
        points=points,
        segment_lengths=tuple(segment_lengths),
        cumulative_lengths=tuple(cumulative_lengths),
        total_length_m=cumulative_lengths[-1],
        segment_projections=tuple(segment_projections),
        segment_headings_degrees=tuple(segment_headings_degrees),
    )


def project_track_position(model: TrackProgressModel, position: TrackPoint) -> TrackProjection:
    """Project a world point onto the nearest track centerline segment."""
    best_segment_index = 0
    best_t = 0.0
    best_distance_squared = float("inf")
    best_nearest_x = model.segment_projections[0][0] if model.segment_projections else model.points[0].x
    best_nearest_z = model.segment_projections[0][1] if model.segment_projections else model.points[0].z
    best_left_normal_x = 0.0
    best_left_normal_z = 1.0
    best_heading_degrees = 0.0
    position_x = position.x
    position_z = position.z

    for index, segment in enumerate(_track_progress_segment_projections(model)):
        start_x, start_z, dx, dz, inverse_length_squared, left_normal_x, left_normal_z, heading_degrees = segment
        t = ((position_x - start_x) * dx + (position_z - start_z) * dz) * inverse_length_squared
        if t <= 0.0:
            t = 0.0
        elif t >= 1.0:
            t = 1.0
        nearest_x = start_x + dx * t
        nearest_z = start_z + dz * t
        distance_x = position_x - nearest_x
        distance_z = position_z - nearest_z
        distance_squared = distance_x * distance_x + distance_z * distance_z
        if distance_squared < best_distance_squared:
            best_segment_index = index
            best_t = t
            best_distance_squared = distance_squared
            best_nearest_x = nearest_x
            best_nearest_z = nearest_z
            best_left_normal_x = left_normal_x
            best_left_normal_z = left_normal_z
            best_heading_degrees = heading_degrees

    segment_length = model.segment_lengths[best_segment_index]
    signed_distance = (position_x - best_nearest_x) * best_left_normal_x + (
        position_z - best_nearest_z
    ) * best_left_normal_z
    progress_distance = model.cumulative_lengths[best_segment_index] + best_t * segment_length
    if model.total_length_m - progress_distance <= 1e-9:
        progress_distance = 0.0
    return TrackProjection(
        position=position,
        nearest_center=TrackPoint(best_nearest_x, best_nearest_z),
        progress_distance_m=progress_distance,
        lap_progress=progress_distance / model.total_length_m,
        signed_distance_to_center_m=signed_distance,
        heading_degrees=best_heading_degrees,
    )


def track_heading_at_distance(model: TrackProgressModel, progress_distance_m: float) -> float:
    """Find the track direction at a distance around the lap."""
    segment_index, _, _ = _segment_at_distance(model=model, progress_distance_m=progress_distance_m)
    if model.segment_headings_degrees:
        return model.segment_headings_degrees[segment_index]
    return track_heading_degrees(
        start=model.points[segment_index], end=model.points[(segment_index + 1) % len(model.points)]
    )


def track_pose_at_distance(model: TrackProgressModel, progress_distance_m: float) -> TrackPose:
    """Find the centerline pose at a distance around the lap."""
    segment_index, t, wrapped_progress = _segment_at_distance(model=model, progress_distance_m=progress_distance_m)
    start = model.points[segment_index]
    end = model.points[(segment_index + 1) % len(model.points)]
    return TrackPose(
        position=TrackPoint(x=start.x + (end.x - start.x) * t, z=start.z + (end.z - start.z) * t),
        heading_degrees=track_heading_degrees(start=start, end=end),
        progress_distance_m=wrapped_progress,
    )


def track_heading_degrees(*, start: TrackPoint, end: TrackPoint) -> float:
    """Compute the direction from one track point to another."""
    return degrees(atan2(end.x - start.x, end.z - start.z))


def heading_error_degrees(*, current_heading_degrees: float, target_heading_degrees: float) -> float:
    """Measure the smallest left/right turn needed to face a target heading."""
    return ((target_heading_degrees - current_heading_degrees + 180.0) % 360.0) - 180.0


def _rebased_track_points(points: tuple[TrackPoint, ...], *, start_line: TrackPoint) -> tuple[TrackPoint, ...]:
    projection_model = build_track_progress_model(points)
    start_projection = project_track_position(projection_model, start_line)
    return _rebased_track_points_at_distance(projection_model, start_projection.progress_distance_m)


def _rebased_track_points_at_distance(model: TrackProgressModel, progress_distance_m: float) -> tuple[TrackPoint, ...]:
    if len(model.points) < 3:
        raise ValueError("a track progress model needs at least three points")
    segment_index, _, _ = _segment_at_distance(model=model, progress_distance_m=progress_distance_m)
    rebased_points = (
        track_pose_at_distance(model, progress_distance_m).position,
        *model.points[segment_index + 1 :],
        *model.points[: segment_index + 1],
    )
    if (
        hypot(
            rebased_points[-1].x - rebased_points[0].x,
            rebased_points[-1].z - rebased_points[0].z,
        )
        <= 1e-9
    ):
        return rebased_points[:-1]
    return rebased_points


def _segment_at_distance(model: TrackProgressModel, progress_distance_m: float) -> tuple[int, float, float]:
    if model.total_length_m <= 0:
        raise ValueError("track progress model total length must be positive")
    wrapped_progress = progress_distance_m % model.total_length_m
    segment_index = max(
        0, min(len(model.segment_lengths) - 1, bisect_right(model.cumulative_lengths, wrapped_progress) - 1)
    )
    segment_start = model.cumulative_lengths[segment_index]
    segment_length = model.segment_lengths[segment_index]
    t = 0.0 if segment_length <= 0 else (wrapped_progress - segment_start) / segment_length
    return segment_index, t, wrapped_progress


def _track_progress_segment_projections(model: TrackProgressModel) -> tuple[TrackProgressSegmentProjection, ...]:
    if model.segment_projections:
        return model.segment_projections
    return build_track_progress_model(model.points).segment_projections
