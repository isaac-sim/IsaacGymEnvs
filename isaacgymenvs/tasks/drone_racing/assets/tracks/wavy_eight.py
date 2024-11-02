from dataclasses import dataclass
from math import radians
from typing import List, Tuple

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ..utils.urdf_utils import (
    cylinder_link,
    cuboid_wireframe_link,
    hollow_cuboid_link,
)
from ...waypoint import Waypoint


@dataclass
class TrackWavyEightOptions:
    file_name: str = "track_wavy_eight"
    track_options: TrackOptions = TrackOptions()
    asset_options: AssetOptions = AssetOptions()
    add_obstacles: bool = False


def create_track_wavy_eight(
    gym: Gym, sim: Sim, options: TrackWavyEightOptions
) -> Tuple[Asset, List[Waypoint]]:
    waypoints = _define_waypoints()

    obstacle_links = []
    obstacle_origins = []
    if options.add_obstacles:
        obstacle_links, obstacle_origins = _define_obstacles()

    asset = create_track_asset(
        options.file_name,
        options.track_options,
        _define_waypoints(),
        obstacle_links,
        obstacle_origins,
        [False] * len(obstacle_links),
        options.asset_options,
        gym,
        sim,
    )
    return asset, waypoints


def _define_waypoints() -> List[Waypoint]:
    r = 8
    gate_size = 1.7
    return [
        Waypoint(
            index=0,
            xyz=[0.0, 0.0, 2.25],
            rpy=[0.0, 0.0, 0.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[r, 0.0, 3.0],
            rpy=[0.0, 0.0, 0.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[2 * r, r, 4.5],
            rpy=[0.0, 0.0, 90.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[r, 2 * r, 4.5],
            rpy=[0.0, 0.0, 180.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[0, r, 6.0],
            rpy=[0.0, 0.0, -90.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=5,
            xyz=[0, -r, 6.0],
            rpy=[0.0, 0.0, -90.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=6,
            xyz=[-r, -2 * r, 4.5],
            rpy=[0.0, 0.0, -180.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[-2 * r, -r, 3.0],
            rpy=[0.0, 0.0, 90.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=8,
            xyz=[-r, 0, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=True,
        ),
        Waypoint(
            index=9,
            xyz=[0, 0, 2.25],
            rpy=[0.0, 0.0, 0.0],
            length_y=gate_size,
            length_z=gate_size,
            gate=False,
        ),
    ]


def _define_obstacles() -> Tuple[List[urdfpy.Link], List[List[float]]]:
    obstacle_links = []
    obstacle_origins = []

    r = 8

    # cuboid wireframes
    obstacle_links.append(cuboid_wireframe_link("obstacle_0", [5.5, 5.5, 5.5], 0.5))
    obstacle_origins.append([0.0, 0.0, 2.75, 0.0, radians(45.0), 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_1", [6, 6, 4], 1.0))
    obstacle_origins.append([2 * r, r, 4.5, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_2", [6, 6, 4], 1.0))
    obstacle_origins.append([-2 * r, -r, 3.0, 0.0, 0.0, 0.0])

    # cylinders
    obstacle_links.append(cylinder_link("obstacle_3", 0.3, 3.0))
    obstacle_origins.append([2 * r, r + 3, 4.5, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_4", 0.3, 3.0))
    obstacle_origins.append([-2 * r, -r - 3, 3.0, 0.0, 0.0, 0.0])

    # hollow cuboids
    obstacle_links.append(hollow_cuboid_link("obstacle_5", 1.0, 2.0, 4.0, 2.0, 4.0))
    obstacle_origins.append([1, 15, 5.0, 0.0, 0.0, radians(45)])

    obstacle_links.append(hollow_cuboid_link("obstacle_6", 1.0, 2.0, 4.0, 2.0, 4.0))
    obstacle_origins.append([-1, -15, 5.0, 0.0, 0.0, radians(45)])

    return obstacle_links, obstacle_origins
