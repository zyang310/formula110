"""Higher-speed preview controller preset for leaderboard racing."""

from dataclasses import replace

from controllers.minimum_viable import MINIMUM_VIABLE_PARAMETERS
from controllers.preview_controller import PreviewController

RACING_NAME: str = "Race Faster"
RACING_COLOR: str = "#f05a47"

# Searched values from artifacts/controller-search/faster-line-v2-ga/checkpoint.json,
# generation 40. Regenerate this block with scripts.controller_training.bake.
RACE_FASTER_PARAMETERS = replace(
    MINIMUM_VIABLE_PARAMETERS,
    heading_steer_gain=0.6461824399091548,
    center_steer_gain=0.1,
    racing_line_offset_ratio=0.5083185108672515,
    racing_line_entry_offset_ratio=0.6485988179706657,
    racing_line_exit_offset_ratio=0.010435384830851038,
    curvature_offset_compensation=0.03391989631932343,
    curvature_heading_compensation=0.024972237497698377,
    line_turn_sensitivity=0.01,
    line_target_slew_per_tick=0.05433520411171578,
    line_clearance_m=2.253163370559477,
    wall_balance_gain=0.04160726063242451,
    curvature_lateral_ratio=1.0,
    straight_target_speed_mps=22.971457980416037,
    corner_target_speed_mps=19.594866917163838,
    steering_speed_reduction=0.006676181231114552,
    front_brake_start_m=10.95292686958372,
    throttle_gain=1.5302277742588817,
    # Carried over; not in the `faster-line-v2` search space.
    yaw_damping_gain=0.10460502116674592,
    phase_aware_racing_line=True,
    pose_invariant_racing_line=True,
    preview_line_compensation=1.0,
    wall_balance_line_compensation=1.0,
    steer_slew_per_tick=0.12814802135925207,
    curvature_heading_degrees=69.68831780256738,
    yaw_speed_reduction=0.0,
    side_slow_start_m=1.3388939039697219,
    side_speed_floor=0.5941676492824329,
    brake_gain=0.17367935069640417,
    maximum_forward_throttle=1.0,
    competitor_racing_steer_fraction=0.30,
    competitor_pass_steer_gain=0.82,
)


def create_controller() -> PreviewController:
    """Create independent smoothing and recovery state for one car."""
    return PreviewController(RACE_FASTER_PARAMETERS)
