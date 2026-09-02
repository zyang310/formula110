"""Human-inspired opposite-turn connector prototype for seed-agnostic tuning."""

from dataclasses import replace

from controllers.preview_controller import PreviewController
from controllers.race_faster import RACE_FASTER_PARAMETERS

RACING_NAME: str = "Race Faster V29"
RACING_COLOR: str = "#ef8a35"

# The detector follows geometry rather than track progress.  It first sees one
# turn enter the preview, then waits for the far preview to switch to the next
# turn.  Once the near preview crosses into that second turn it applies the
# human-inspired rotate -> coast/alignment -> countersteer sequence.
RACE_FASTER_V29_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    transition_drift_minimum_speed_mps=22.0,
    transition_drift_preview_curvature=0.04,
    transition_drift_trigger_curvature=0.01,
    transition_drift_preview_seconds=0.75,
    transition_drift_target_speed_bonus_mps=0.50,
    transition_drift_brake=0.05,
    transition_drift_steer=1.0,
    transition_drift_steer_slew_per_tick=0.30,
    transition_drift_pulse_seconds=1.0 / 60.0,
    transition_drift_coast_max_seconds=0.0,
    transition_drift_minimum_heading_error_degrees=12.0,
    transition_drift_alignment_heading_degrees=3.0,
    transition_drift_countersteer=0.75,
    transition_drift_countersteer_max_seconds=2.0 / 60.0,
    transition_drift_alignment_yaw_rate_degrees_per_s=60.0,
    transition_drift_cooldown_seconds=2.50,
)


def create_controller() -> PreviewController:
    """Create independent smoothing, recovery, and connector state for one car."""
    return PreviewController(RACE_FASTER_V29_PARAMETERS)
