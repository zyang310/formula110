"""Pure track geometry, layout constants, and centerline sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from math import hypot
from typing import Literal

TrackComplexity = Literal["beginner", "easy", "intermediate", "advanced", "expert"]


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """Two-dimensional point on the racing track layout."""

    x: float
    z: float
    label: str = ""


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """Line segment between two track centerline points."""

    start: TrackPoint
    end: TrackPoint

    @property
    def length(self) -> float:
        """Measure this centerline segment in world units."""
        return distance_between(self.start, self.end)


@dataclass(frozen=True, slots=True)
class TrackBounds:
    """Axis-aligned bounds of a track layout in X/Z space."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        """Measure the X span of these bounds."""
        return self.max_x - self.min_x

    @property
    def length(self) -> float:
        """Measure the Z span of these bounds."""
        return self.max_z - self.min_z


@dataclass(frozen=True, slots=True)
class TrackLayout:
    """Named pure centerline layout used by racing scenes."""

    track_id: str
    points: tuple[TrackPoint, ...]
    start_position: TrackPoint
    complexity: TrackComplexity = "intermediate"


TRACK_SCALE = 2.0
TRACK_LAYOUT_SCALE = 2.8
NOMINAL_CAR_WIDTH = 1.0 * TRACK_SCALE
TRACK_WIDTH = (2.3 * TRACK_SCALE) + NOMINAL_CAR_WIDTH
TRACK_ID_MUGELLO_SHORT = "mugello-short"
TRACK_ID_MUGELLO_SHORT_WIDE = "mugello-short-wide"
TRACK_ID_MUGELLO_SHORT_LONG = "mugello-short-long"
TRACK_ID_CAROLINA_LOOP = "carolina-loop"
TRACK_ID_STADIUM_LOOP = "stadium-loop"
TRACK_ID_HARBOR_LOOP = "harbor-loop"
TRACK_ID_DOGWOOD_LOOP = "dogwood-loop"
TRACK_ID_BLUE_RIDGE_LOOP = "blue-ridge-loop"
TRACK_ID_PINE_SWITCHBACKS = "pine-switchbacks"


def _scaled_track_point(x: float, z: float, label: str = "") -> TrackPoint:
    # Mirror the traced map points so clockwise laps leave the start line toward San Donato.
    return TrackPoint(-x * TRACK_LAYOUT_SCALE, z * TRACK_LAYOUT_SCALE, label)


def _layout_track_point(x: float, z: float, label: str = "") -> TrackPoint:
    return TrackPoint(x * TRACK_LAYOUT_SCALE, z * TRACK_LAYOUT_SCALE, label)


def _layout_points(
    name: str,
    coordinates: tuple[tuple[float, float], ...],
) -> tuple[TrackPoint, ...]:
    return tuple(_layout_track_point(x, z, f"{name} sector {index + 1}") for index, (x, z) in enumerate(coordinates))


def _layout_start_position(points: tuple[TrackPoint, ...], label: str) -> TrackPoint:
    return TrackPoint(points[0].x, points[0].z, label)


MAIN_STRAIGHT_ENTRY = _scaled_track_point(5.0, -6.2, "Main straight entry")
MAIN_STRAIGHT = _scaled_track_point(1.0, -6.1, "Main straight")
START_STRAIGHT = _scaled_track_point(-5.8, -5.9, "Start straight")
START_POSITION = TrackPoint(
    x=(MAIN_STRAIGHT.x + START_STRAIGHT.x) / 2,
    z=(MAIN_STRAIGHT.z + START_STRAIGHT.z) / 2,
    label="Start grid",
)

MUGELLO_SHORT_LAYOUT: tuple[TrackPoint, ...] = (
    MAIN_STRAIGHT_ENTRY,
    MAIN_STRAIGHT,
    START_STRAIGHT,
    _scaled_track_point(-12.4, -5.8, "San Donato approach"),
    _scaled_track_point(-16.2, -5.2, "San Donato braking"),
    _scaled_track_point(-17.6, -3.8, "San Donato"),
    _scaled_track_point(-16.8, -2.2, "San Donato exit"),
    _scaled_track_point(-14.5, -1.6, "Luco"),
    _scaled_track_point(-13.0, -0.6, "Luco climb"),
    _scaled_track_point(-12.7, 2.3, "Poggiosecco"),
    _scaled_track_point(-10.8, 3.8, "Poggiosecco exit"),
    _scaled_track_point(-6.2, 3.0, "Materassi"),
    _scaled_track_point(-2.2, 2.3, "Materassi exit"),
    _scaled_track_point(-0.2, 4.1, "Borgo San Lorenzo"),
    _scaled_track_point(1.6, 4.3, "Borgo crest"),
    _scaled_track_point(4.9, 3.4, "Casanova approach"),
    _scaled_track_point(7.5, 2.3, "Casanova"),
    _scaled_track_point(8.8, -0.6, "Casanova exit"),
    _scaled_track_point(8.4, -3.5, "Return bend"),
    _scaled_track_point(7.0, -5.5, "Final return"),
)


def _scaled_layout_about_start(
    points: tuple[TrackPoint, ...],
    *,
    x_scale: float,
    z_scale: float,
) -> tuple[TrackPoint, ...]:
    return tuple(
        TrackPoint(
            x=START_POSITION.x + (point.x - START_POSITION.x) * x_scale,
            z=START_POSITION.z + (point.z - START_POSITION.z) * z_scale,
            label=point.label,
        )
        for point in points
    )


MUGELLO_SHORT_WIDE_LAYOUT = _scaled_layout_about_start(
    MUGELLO_SHORT_LAYOUT,
    x_scale=1.06,
    z_scale=0.96,
)
MUGELLO_SHORT_LONG_LAYOUT = _scaled_layout_about_start(
    MUGELLO_SHORT_LAYOUT,
    x_scale=0.94,
    z_scale=1.06,
)

CAROLINA_LOOP_LAYOUT: tuple[TrackPoint, ...] = (
    _layout_track_point(-3.6, -8.6, "Start straight entry"),
    _layout_track_point(1.8, -8.6, "Start straight"),
    _layout_track_point(7.1, -7.9, "Dogwood approach"),
    _layout_track_point(11.4, -5.7, "Dogwood"),
    _layout_track_point(13.9, -2.1, "Ridge bend"),
    _layout_track_point(14.3, 2.1, "East sweeper"),
    _layout_track_point(12.1, 6.1, "East sweeper exit"),
    _layout_track_point(8.2, 8.2, "Back straight entry"),
    _layout_track_point(3.6, 8.9, "Back straight"),
    _layout_track_point(-1.1, 7.9, "Blue Ridge approach"),
    _layout_track_point(-5.4, 6.1, "Blue Ridge"),
    _layout_track_point(-8.9, 3.2, "Piedmont descent"),
    _layout_track_point(-10.7, 0.0, "Piedmont bend"),
    _layout_track_point(-12.9, -1.4, "Pine hairpin entry"),
    _layout_track_point(-15.0, -3.6, "Pine hairpin"),
    _layout_track_point(-13.2, -6.4, "Pine hairpin exit"),
    _layout_track_point(-8.9, -8.2, "Final turn"),
)
CAROLINA_LOOP_START_POSITION = TrackPoint(
    x=(CAROLINA_LOOP_LAYOUT[0].x + CAROLINA_LOOP_LAYOUT[1].x) / 2,
    z=(CAROLINA_LOOP_LAYOUT[0].z + CAROLINA_LOOP_LAYOUT[1].z) / 2,
    label="Carolina Loop start grid",
)

# These five circuits are precomputed asymmetric radial layouts. Keeping their
# coordinates explicit avoids platform-dependent trigonometry in simulator
# geometry while progressing from two chicanes to dense mixed-direction turns.
STADIUM_LOOP_LAYOUT = _layout_points(
    "Stadium Loop",
    (
        (0.0, -9.0),
        (5.358, -8.315),
        (9.306, -5.982),
        (10.089, -2.686),
        (9.8, 0.0),
        (11.382, 3.031),
        (10.493, 6.746),
        (5.358, 8.315),
        (0.0, 9.0),
        (-4.929, 7.65),
        (-7.524, 4.837),
        (-10.865, 2.893),
        (-14.56, 0.0),
        (-12.934, -3.444),
        (-9.899, -6.364),
        (-5.358, -8.315),
    ),
)
STADIUM_LOOP_START_POSITION = _layout_start_position(STADIUM_LOOP_LAYOUT, "Stadium Loop start grid")

HARBOR_LOOP_LAYOUT = _layout_points(
    "Harbor Loop",
    (
        (0.0, -11.0),
        (5.582, -9.923),
        (8.96, -6.91),
        (10.011, -3.74),
        (14.063, -1.605),
        (18.081, 2.063),
        (14.722, 5.5),
        (10.053, 7.752),
        (4.303, 7.649),
        (0.0, 9.02),
        (-6.163, 10.957),
        (-10.927, 8.426),
        (-12.956, 4.84),
        (-11.719, 1.337),
        (-13.059, -1.49),
        (-15.017, -5.61),
        (-11.583, -8.932),
        (-5.814, -10.337),
    ),
)
HARBOR_LOOP_START_POSITION = _layout_start_position(HARBOR_LOOP_LAYOUT, "Harbor Loop start grid")

DOGWOOD_LOOP_LAYOUT = _layout_points(
    "Dogwood Loop",
    (
        (0.0, -13.0),
        (5.297, -11.725),
        (8.434, -8.53),
        (9.976, -5.619),
        (14.918, -4.428),
        (21.776, -2.035),
        (19.796, 1.85),
        (16.373, 4.86),
        (10.883, 6.129),
        (8.65, 8.749),
        (5.973, 13.222),
        (0.0, 13.0),
        (-4.958, 10.977),
        (-6.92, 6.999),
        (-11.487, 6.47),
        (-20.376, 6.048),
        (-19.796, 1.85),
        (-18.213, -1.702),
        (-12.735, -3.78),
        (-12.999, -7.321),
        (-11.245, -11.374),
        (-5.635, -12.473),
    ),
)
DOGWOOD_LOOP_START_POSITION = _layout_start_position(DOGWOOD_LOOP_LAYOUT, "Dogwood Loop start grid")

BLUE_RIDGE_LOOP_LAYOUT = _layout_points(
    "Blue Ridge Loop",
    (
        (0.0, -16.0),
        (5.284, -14.292),
        (8.253, -10.484),
        (9.867, -7.425),
        (15.406, -7.089),
        (25.133, -6.355),
        (23.825, -1.929),
        (20.966, 1.697),
        (15.259, 3.858),
        (15.011, 6.908),
        (17.188, 12.934),
        (11.153, 14.167),
        (5.169, 13.982),
        (0.0, 10.24),
        (-4.595, 12.428),
        (-12.715, 16.151),
        (-15.915, 11.976),
        (-16.986, 7.817),
        (-14.811, 3.745),
        (-17.631, 1.427),
        (-26.208, -2.121),
        (-22.44, -5.674),
        (-18.171, -8.362),
        (-11.14, -8.383),
        (-9.369, -11.901),
        (-5.973, -16.156),
    ),
)
BLUE_RIDGE_LOOP_START_POSITION = _layout_start_position(BLUE_RIDGE_LOOP_LAYOUT, "Blue Ridge Loop start grid")

PINE_SWITCHBACKS_LAYOUT = _layout_points(
    "Pine Switchbacks",
    (
        (0.0, -19.0),
        (5.123, -16.355),
        (7.744, -11.803),
        (9.546, -8.915),
        (16.23, -9.917),
        (27.644, -10.83),
        (25.564, -5.636),
        (20.05, -1.43),
        (17.265, 1.231),
        (22.369, 4.932),
        (26.189, 10.26),
        (18.727, 11.442),
        (10.533, 9.838),
        (6.605, 10.067),
        (4.657, 14.868),
        (0.0, 22.04),
        (-5.822, 18.585),
        (-8.428, 12.844),
        (-9.875, 9.223),
        (-17.063, 10.425),
        (-26.674, 10.45),
        (-24.499, 5.402),
        (-18.379, 1.311),
        (-17.265, -1.231),
        (-22.901, -5.049),
        (-27.159, -10.64),
        (-19.976, -12.205),
        (-11.521, -10.76),
        (-8.883, -13.539),
        (-6.054, -19.328),
    ),
)
PINE_SWITCHBACKS_START_POSITION = _layout_start_position(
    PINE_SWITCHBACKS_LAYOUT,
    "Pine Switchbacks start grid",
)
TRACK_LAYOUTS = (
    TrackLayout(
        track_id=TRACK_ID_MUGELLO_SHORT,
        points=MUGELLO_SHORT_LAYOUT,
        start_position=START_POSITION,
        complexity="advanced",
    ),
    TrackLayout(
        track_id=TRACK_ID_MUGELLO_SHORT_WIDE,
        points=MUGELLO_SHORT_WIDE_LAYOUT,
        start_position=START_POSITION,
        complexity="advanced",
    ),
    TrackLayout(
        track_id=TRACK_ID_MUGELLO_SHORT_LONG,
        points=MUGELLO_SHORT_LONG_LAYOUT,
        start_position=START_POSITION,
        complexity="advanced",
    ),
    TrackLayout(
        track_id=TRACK_ID_CAROLINA_LOOP,
        points=CAROLINA_LOOP_LAYOUT,
        start_position=CAROLINA_LOOP_START_POSITION,
        complexity="intermediate",
    ),
    TrackLayout(
        track_id=TRACK_ID_STADIUM_LOOP,
        points=STADIUM_LOOP_LAYOUT,
        start_position=STADIUM_LOOP_START_POSITION,
        complexity="beginner",
    ),
    TrackLayout(
        track_id=TRACK_ID_HARBOR_LOOP,
        points=HARBOR_LOOP_LAYOUT,
        start_position=HARBOR_LOOP_START_POSITION,
        complexity="easy",
    ),
    TrackLayout(
        track_id=TRACK_ID_DOGWOOD_LOOP,
        points=DOGWOOD_LOOP_LAYOUT,
        start_position=DOGWOOD_LOOP_START_POSITION,
        complexity="intermediate",
    ),
    TrackLayout(
        track_id=TRACK_ID_BLUE_RIDGE_LOOP,
        points=BLUE_RIDGE_LOOP_LAYOUT,
        start_position=BLUE_RIDGE_LOOP_START_POSITION,
        complexity="advanced",
    ),
    TrackLayout(
        track_id=TRACK_ID_PINE_SWITCHBACKS,
        points=PINE_SWITCHBACKS_LAYOUT,
        start_position=PINE_SWITCHBACKS_START_POSITION,
        complexity="expert",
    ),
)


def track_layout_ids() -> tuple[str, ...]:
    """Return supported track layout ids."""
    return tuple(layout.track_id for layout in TRACK_LAYOUTS)


def track_layout_by_id(track_id: str) -> TrackLayout:
    """Return a named pure track layout."""
    for layout in TRACK_LAYOUTS:
        if layout.track_id == track_id:
            return layout
    valid_ids = ", ".join(track_layout_ids())
    raise ValueError(f"unknown track layout: {track_id}; expected one of {valid_ids}")


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to an inclusive numeric range."""
    return min(max(value, low), high)


def distance_between(start: TrackPoint, end: TrackPoint) -> float:
    """Return the X/Z distance between two track points."""
    return hypot(end.x - start.x, end.z - start.z)


def closed_track_points(points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT) -> tuple[TrackPoint, ...]:
    """Return centerline points with the starting point repeated at the end."""
    if len(points) < 3:
        raise ValueError("a racing track needs at least three centerline points")
    return (*points, points[0])


def track_segments(points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT) -> tuple[TrackSegment, ...]:
    """Return centerline segments, including the closing segment."""
    closed_points = closed_track_points(points)
    return tuple(TrackSegment(start=start, end=end) for start, end in pairwise(closed_points))


def total_track_length(points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT) -> float:
    """Return the full closed-loop centerline length."""
    return sum(segment.length for segment in track_segments(points))


@lru_cache(maxsize=16)
def sampled_track_centerline(
    points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT,
    *,
    samples_per_segment: int = 8,
) -> tuple[TrackPoint, ...]:
    """Return smooth Catmull-Rom samples around the track centerline."""
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be at least one")
    if len(points) < 4:
        raise ValueError("a smooth racing track needs at least four centerline points")

    samples: list[TrackPoint] = []
    for index, point in enumerate(points):
        previous_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        after_next_point = points[(index + 2) % len(points)]

        for sample_index in range(samples_per_segment):
            t = sample_index / samples_per_segment
            samples.append(
                TrackPoint(
                    x=_catmull_rom(previous_point.x, point.x, next_point.x, after_next_point.x, t),
                    z=_catmull_rom(previous_point.z, point.z, next_point.z, after_next_point.z, t),
                    label=point.label if sample_index == 0 else "",
                )
            )
    return tuple(samples)


def track_bounds(
    points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT,
    *,
    margin: float = TRACK_WIDTH,
) -> TrackBounds:
    """Return X/Z bounds for a set of track points plus margin."""
    xs = tuple(point.x for point in points)
    zs = tuple(point.z for point in points)
    return TrackBounds(
        min_x=min(xs) - margin,
        max_x=max(xs) + margin,
        min_z=min(zs) - margin,
        max_z=max(zs) + margin,
    )


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
