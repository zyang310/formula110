"""Speedmaxxing variant of :mod:`controllers.race_faster`.

This preset deliberately accepts barrier contact and slower lap times in order
to maximize the sustained speed reached on the long corridor.  Keep
``race_faster`` as the balanced lap-time controller.
"""

from dataclasses import replace

from controllers.preview_controller import PreviewController
from controllers.race_faster import RACE_FASTER_PARAMETERS

RACING_NAME: str = "Race Speedmax"
RACING_COLOR: str = "#f7b32b"

# Searched values from
# artifacts/controller-search/faster-line-v26-ga/generations/generation-006.json.
RACE_SPEEDMAX_PARAMETERS = replace(
    RACE_FASTER_PARAMETERS,
    long_straight_minimum_duration_s=0.34823084582388436,
    long_straight_maximum_local_curvature=0.005274529604018754,
    long_straight_speed_bonus_seconds=0.6414503014138803,
    long_straight_target_speed_bonus_mps=25.0,
    straight_target_speed_mps=25.48204087398701,
    steering_speed_reduction=0.030718800196918146,
    front_brake_start_m=11.373183325660271,
    front_stop_m=2.4085355195849933,
)


def create_controller() -> PreviewController:
    """Create independent smoothing and recovery state for one car."""
    return PreviewController(RACE_SPEEDMAX_PARAMETERS)
