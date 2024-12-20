from dataclasses import dataclass, field
from typing import List


@dataclass
class TrackOptions:
    # flag to enable debugging visualization
    enable_debug_visualization: bool = False

    # difference of gate outer and inner length
    gate_size: float = 0.3

    # x-axis length of the hollow cuboid representing the gates
    gate_length_x: float = 0.15

    # gate color
    gate_color: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.3, 1.0])

    # additional obstacle color
    additional_obstacle_color: List[float] = field(
        default_factory=lambda: [0.0, 0.75, 1.0, 1.0]
    )

    # radius for the cylinder used to show line segments in debug
    debug_cylinder_radius: float = 0.05

    # waypoint x-axis length in debug
    debug_waypoint_length_x: float = 0.025

    # color of debug visualization
    debug_visual_color: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 0.0, 1.0]
    )
