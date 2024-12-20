from dataclasses import dataclass
from math import sqrt, radians
from typing import List, Tuple

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ..utils.urdf_utils import cuboid_link
from ...waypoint import Waypoint


@dataclass
class TrackPlanarCircleOptions:
    file_name: str = "track_planar_circle"
    track_options: TrackOptions = TrackOptions()
    asset_options: AssetOptions = AssetOptions()
    add_obstacles: bool = False


def create_track_planar_circle(
    gym: Gym, sim: Sim, options: TrackPlanarCircleOptions
) -> Tuple[Asset, List[Waypoint]]:
    """
    Create a track with obstacles shaped as a planar circle.

    Args:
        gym: returned by ``acquire_gym``.
        sim: simulation handle.
        options: options for the asset, and importing.

    Returns:
        - An asset object as the return of calling ``load_asset``.
        - A list of ``Waypoint`` instances.
    """
    waypoints = _define_waypoints()

    obstacle_links = []
    obstacle_origins = []
    if options.add_obstacles:
        obstacle_links, obstacle_origins = _define_obstacles()

    asset = create_track_asset(
        options.file_name,
        options.track_options,
        waypoints,
        obstacle_links,
        obstacle_origins,
        [False] * len(obstacle_links),
        options.asset_options,
        gym,
        sim,
    )
    return asset, waypoints


def _define_waypoints() -> List[Waypoint]:
    r = 15
    l = r / sqrt(2)
    h = 1.5
    g = 1.75

    return [
        Waypoint(
            index=0,
            xyz=[0.0, r, h],
            rpy=[0.0, 0.0, 0.0],
            length_y=g,
            length_z=g,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[l, l, h],
            rpy=[0.0, 0.0, -45.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[r, 0, h],
            rpy=[0.0, 0.0, -90.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[l, -l, h],
            rpy=[0.0, 0.0, -135.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[0, -r, h],
            rpy=[0.0, 0.0, -180.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
        Waypoint(
            index=5,
            xyz=[-l, -l, h],
            rpy=[0.0, 0.0, -225.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
        Waypoint(
            index=6,
            xyz=[-r, 0, h],
            rpy=[0.0, 0.0, -270.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[-l, l, h],
            rpy=[0.0, 0.0, -315.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
        Waypoint(
            index=8,
            xyz=[0, r, h],
            rpy=[0.0, 0.0, 0.0],
            length_y=g,
            length_z=g,
            gate=True,
        ),
    ]


def _define_obstacles() -> Tuple[List[urdfpy.Link], List[List[float]]]:
    obstacle_links = []
    obstacle_origins = []

    # cuboids
    obstacle_links.append(cuboid_link("obstacle_0", [0.5, 38.0, 1.5]))
    obstacle_origins.append([0.0, 0.0, 1.5, radians(-2.5), 0.0, radians(-22.5)])

    obstacle_links.append(cuboid_link("obstacle_1", [0.5, 38.0, 1.5]))
    obstacle_origins.append([0.0, 0.0, 1.5, radians(+2.5), 0.0, radians(-67.5)])

    obstacle_links.append(cuboid_link("obstacle_2", [0.5, 30.0, 5.0]))
    obstacle_origins.append([0.0, 0.0, 1.5, 0.0, 0.0, radians(67.5)])

    return obstacle_links, obstacle_origins
