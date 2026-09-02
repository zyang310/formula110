#!/usr/bin/env python3
"""Generate compact progress-indexed policy data for the hybrid controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from racing.race.progress import (  # noqa: E402
    default_track_progress_model,
    heading_error_degrees,
    track_pose_at_distance,
)
from racing.track.spatial import track_forward_vector  # noqa: E402
from racing.track.world import TRACK_WIDTH, TrackPoint  # noqa: E402

DEFAULT_OUTPUT = SOURCE_ROOT / "controllers" / "hybrid_track_policy_data.py"
LOOKAHEAD_DISTANCES_M = (4.0, 9.0, 16.0)


def build_rows(bin_count: int) -> tuple[tuple[float, float, float, float, float, float, float, float, float], ...]:
    model = default_track_progress_model()
    step_m = model.total_length_m / bin_count
    rows: list[tuple[float, float, float, float, float, float, float, float, float]] = []
    for index in range(bin_count):
        progress_m = index * step_m
        pose = track_pose_at_distance(model, progress_m)
        back_heading = track_pose_at_distance(model, progress_m - 5.0).heading_degrees
        forward_heading = track_pose_at_distance(model, progress_m + 5.0).heading_degrees
        curvature = (
            heading_error_degrees(current_heading_degrees=back_heading, target_heading_degrees=forward_heading) / 10.0
        )
        lookahead_offsets = tuple(
            _right_offset(
                origin=pose.position,
                heading_degrees=pose.heading_degrees,
                target=track_pose_at_distance(model, progress_m + lookahead_m).position,
            )
            for lookahead_m in LOOKAHEAD_DISTANCES_M
        )
        abs_curvature = abs(curvature)
        future_abs_curvature = max(
            abs(
                heading_error_degrees(
                    current_heading_degrees=track_pose_at_distance(model, progress_m + offset_m - 5.0).heading_degrees,
                    target_heading_degrees=track_pose_at_distance(model, progress_m + offset_m + 5.0).heading_degrees,
                )
                / 10.0
            )
            for offset_m in (0.0, 8.0, 16.0, 24.0)
        )
        target_speed = max(5.2, min(17.0, 16.5 - future_abs_curvature * 3.8 - abs_curvature * 1.2))
        target_offset = max(-1.8, min(1.8, -_sign(curvature) * min(1.45, 0.42 + abs_curvature * 0.50)))
        steer_ff = max(-0.34, min(0.34, curvature * 0.11))
        brake_bias = max(0.0, min(1.0, future_abs_curvature / 2.4))
        rows.append(
            (
                round(progress_m, 3),
                round(pose.heading_degrees, 3),
                round(curvature, 5),
                round(lookahead_offsets[0], 4),
                round(lookahead_offsets[1], 4),
                round(lookahead_offsets[2], 4),
                round(target_speed, 3),
                round(target_offset, 3),
                round(steer_ff + brake_bias * 0.02 * _sign(curvature), 4),
            )
        )
    return tuple(rows)


def render_python(rows: tuple[tuple[float, float, float, float, float, float, float, float, float], ...]) -> str:
    model = default_track_progress_model()
    labels = tuple((point.label, point.x, point.z) for point in model.points if point.label)
    text = [
        '"""Generated compact track policy data for controllers.hybrid_track_policy."""',
        "",
        "from __future__ import annotations",
        "",
        "TRACK_TOTAL_LENGTH_M: float = " + repr(round(model.total_length_m, 6)),
        "TRACK_WIDTH_M: float = " + repr(TRACK_WIDTH),
        "LOOKAHEAD_DISTANCES_M: tuple[float, float, float] = " + repr(LOOKAHEAD_DISTANCES_M),
        "POLICY_ROW_FIELDS: tuple[str, ...] = (",
        '    "progress_m",',
        '    "heading_degrees",',
        '    "curvature",',
        '    "lookahead_4m",',
        '    "lookahead_9m",',
        '    "lookahead_16m",',
        '    "target_speed_mps",',
        '    "target_offset_m",',
        '    "steer_feedforward",',
        ")",
        "POLICY_ROWS: tuple[tuple[float, float, float, float, float, float, float, float, float], ...] = (",
    ]
    text.extend(f"    {row!r}," for row in rows)
    text.extend(
        [
            ")",
            "TURN_LABELS: tuple[tuple[str, float, float], ...] = (",
            *(f"    {label!r}," for label in labels),
            ")",
            "DEFAULT_GAINS: dict[str, float] = {",
            '    "heading_gain": 0.010,',
            '    "center_gain": 0.17528759,',
            '    "lookahead_gain": 0.070915077,',
            '    "yaw_damping_gain": 0.0034512221,',
            '    "speed_kp": 0.20662326,',
            '    "speed_brake_kp": 0.12363828,',
            '    "hard_brake_speed_error": 4.0549039,',
            '    "large_heading_error_degrees": 34.266437,',
            '    "front_wall_emergency_m": 2.1578574,',
            '    "side_wall_bias_gain": 0.034274811,',
            '    "stuck_speed_mps": 0.30,',
            '    "stuck_seconds": 1.5958355,',
            '    "target_speed_scale": 1.0,',
            '    "target_speed_bias_mps": 0.0,',
            '    "curve_speed_penalty_scale": 0.0,',
            '    "straight_speed_bonus_mps": 0.0,',
            '    "straight_curvature_threshold": 0.18,',
            '    "min_target_speed_mps": 4.8,',
            '    "max_target_speed_mps": 20.0,',
            '    "front_wall_speed_scale": 2.6,',
            '    "large_heading_target_speed_mps": 4.0,',
            "}",
            "",
        ]
    )
    return "\n".join(text)


def write_data(output: Path, *, bin_count: int) -> Path:
    rows = build_rows(bin_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_python(rows), encoding="utf-8")
    return output.resolve()


def _right_offset(*, origin: TrackPoint, heading_degrees: float, target: TrackPoint) -> float:
    forward_x, forward_z = track_forward_vector(heading_degrees)
    right_x, right_z = forward_z, -forward_x
    return (target.x - origin.x) * right_x + (target.z - origin.z) * right_z


def _sign(value: float) -> float:
    if abs(value) < 1e-6:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=128, help="progress bins around the fixed track")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Python artifact to write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.bins) < 24:
        raise SystemExit("error: --bins must be at least 24")
    output = write_data(args.output, bin_count=int(args.bins))
    row_count = len(build_rows(int(args.bins)))
    print(f"wrote {row_count} policy rows to {output}")


if __name__ == "__main__":
    main()
