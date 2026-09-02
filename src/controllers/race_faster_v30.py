"""Human-inspired same-direction yaw-carry prototype for seed-agnostic tuning."""

from dataclasses import replace

from controllers.preview_controller import PreviewController
from controllers.race_faster import RACE_FASTER_PARAMETERS

RACING_NAME: str = "Race Faster V30"
RACING_COLOR: str = "#8f63d2"

# The new recording never countersteers.  After the short rotation pulse it
# carries yaw with neutral steering, adds one bounded same-direction hold,
# settles neutrally, and then returns normal steering while preserving a speed
# bonus through the following long edge.  These values are the fastest clean
# active candidate from the corrected 33-seed round-three gate.  It remains an
# experiment because it is slower than the promoted controller.
RACE_FASTER_V30_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    transition_drift_minimum_speed_mps=22.0,
    transition_drift_preview_curvature=0.04,
    transition_drift_trigger_curvature=0.01,
    transition_drift_preview_seconds=0.75,
    transition_drift_target_speed_bonus_mps=0.7033277289643927,
    transition_drift_brake=0.04476743877132456,
    transition_drift_steer=1.0,
    transition_drift_steer_slew_per_tick=0.18018771047373464,
    transition_drift_pulse_seconds=0.025218336363338624,
    transition_drift_coast_max_seconds=0.10667069965739279,
    transition_drift_minimum_heading_error_degrees=0.0,
    transition_drift_alignment_heading_degrees=0.0,
    transition_drift_countersteer=0.0,
    transition_drift_countersteer_max_seconds=0.0,
    transition_drift_alignment_yaw_rate_degrees_per_s=71.7398278851807,
    transition_drift_same_direction_hold_trigger_yaw_rate_degrees_per_s=152.03266492492452,
    transition_drift_same_direction_hold=0.2435588215255723,
    transition_drift_same_direction_hold_seconds=0.09022005927122324,
    transition_drift_settle_max_seconds=0.029004230589410096,
    transition_drift_edge_acceleration_seconds=0.4137185579369881,
    transition_drift_cooldown_seconds=2.50,
)


def create_controller() -> PreviewController:
    """Create independent smoothing, recovery, and connector state for one car."""
    return PreviewController(RACE_FASTER_V30_PARAMETERS)
