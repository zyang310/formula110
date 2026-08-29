"""Conservative preview controller preset for the minimum grading gates."""

from controllers.preview_controller import ControllerParameters, PreviewController

RACING_NAME: str = "Minimum Viable"
RACING_COLOR: str = "#2aa876"

MINIMUM_VIABLE_PARAMETERS = ControllerParameters(
    heading_steer_gain=0.9682219507778963,
    center_steer_gain=0.5187553115091447,
    yaw_damping_gain=0.18620657234052151,
    curvature_heading_degrees=49.18888577078214,
    curvature_lateral_ratio=0.7,
    straight_target_speed_mps=12.39023338154619,
    corner_target_speed_mps=7.368777126356012,
    steering_speed_reduction=0.2,
    yaw_speed_reduction=0.20305859473851998,
    front_brake_start_m=7.059982695682738,
    brake_gain=0.2356494938573081,
    maximum_forward_throttle=0.9549422639983814,
)


def create_controller() -> PreviewController:
    """Create independent smoothing and recovery state for one car."""
    return PreviewController(MINIMUM_VIABLE_PARAMETERS)
