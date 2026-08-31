"""Higher-speed preview controller preset for leaderboard racing."""

from dataclasses import replace

from controllers.minimum_viable import MINIMUM_VIABLE_PARAMETERS
from controllers.preview_controller import PreviewController

RACING_NAME: str = "Race Faster"
RACING_COLOR: str = "#f05a47"

# Searched values from artifacts/controller-search/faster-line-v25-ga/generations/generation-010.json, generation 10.
RACE_FASTER_PARAMETERS = replace(
    MINIMUM_VIABLE_PARAMETERS,
    long_straight_minimum_duration_s=0.27260799194145635,
    long_straight_maximum_local_curvature=0.004555684426358618,
    long_straight_speed_bonus_seconds=0.2357936093404097,
    long_straight_target_speed_bonus_mps=0.5547473868287458,
    startup_drift_brake=0.4124807290270464,
    startup_drift_trigger_front_m=10.489831198158283,
    startup_drift_minimum_steer=0.2523921041302151,
    startup_drift_pulse_seconds=0.08162033115099354,
    startup_drift_steer_gain=1.4581016514932057,
    # Carried over; not in this search space.
    heading_steer_gain=0.05,
    center_steer_gain=0.059654446225543084,
    yaw_damping_gain=0.06509264094100722,
    racing_line_offset_ratio=0.4433385662654602,
    racing_line_entry_offset_ratio=0.95,
    phase_aware_racing_line=True,
    pose_invariant_racing_line=True,
    curvature_offset_compensation=0.03391989631932343,
    curvature_heading_compensation=0.024972237497698377,
    preview_line_compensation=1.0,
    wall_balance_line_compensation=1.0,
    maximum_racing_line_offset_ratio=0.95,
    pose_invariant_speed_curvature=True,
    line_turn_sensitivity=0.002,
    line_target_slew_per_tick=0.027653898653721035,
    line_target_release_per_tick=0.25,
    sweeper_minimum_duration_s=2.148433439505907,
    sweeper_speed_hold_seconds=0.3436007106643762,
    sweeper_target_speed_bonus_mps=2.108572238686414,
    sweeper_preview_minimum_far_curvature=0.10670762416340707,
    sweeper_preview_maximum_far_curvature=0.1454324757125199,
    sweeper_preview_speed_hold_seconds=1.1508816075990376,
    sweeper_preview_target_speed_bonus_mps=1.5295216325124745,
    line_clearance_m=0.06461922583329455,
    wall_balance_gain=0.0,
    steer_slew_per_tick=0.12814802135925207,
    curvature_heading_degrees=114.87716155360224,
    curvature_lateral_ratio=3.135466656589882,
    straight_target_speed_mps=25.075973286710095,
    corner_target_speed_mps=13.74405551454704,
    startup_speed_cap_mps=22.773938527649268,
    startup_speed_cap_seconds=1.9925306464172983,
    startup_drift_window_seconds=3.25,
    steering_speed_reduction=0.029956190545059522,
    yaw_speed_reduction=0.0,
    front_brake_start_m=11.91938757401338,
    front_stop_m=2.0365804051333773,
    side_slow_start_m=0.6512447294873163,
    side_speed_floor=0.5833526497921152,
    throttle_gain=5.179198594633248,
    brake_gain=0.17367935069640417,
    maximum_forward_throttle=1.0,
    avoid_front_wall_m=3.174413899598674,
    avoid_diagonal_wall_m=0.5639521581352487,
    avoid_side_wall_m=0.6,
    avoid_speed_mps=3.22674036861771,
    avoid_steer_gain=0.9510593659721247,
    competitor_racing_steer_fraction=0.3,
    competitor_pass_steer_gain=0.82,
)


def create_controller() -> PreviewController:
    """Create independent smoothing and recovery state for one car."""
    return PreviewController(RACE_FASTER_PARAMETERS)
