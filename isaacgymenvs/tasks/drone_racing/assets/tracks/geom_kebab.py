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
    sphere_link,
    cuboid_link,
)
from ...waypoint import Waypoint


@dataclass
class TrackGeomKebabOptions:
    file_name: str = "track_geom_kebab"
    track_options: TrackOptions = TrackOptions()
    asset_options: AssetOptions = AssetOptions()
    add_obstacles: bool = True


def create_track_geom_kebab(
    gym: Gym, sim: Sim, options: TrackGeomKebabOptions
) -> Tuple[Asset, List[Waypoint]]:
    """
    Create a track that is almost straight with obstacles in between.

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
    return [
        Waypoint(
            index=0,
            xyz=[-18.0, 0.0, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.8,
            length_z=1.8,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[-10.0, 1.0, 1.6],
            rpy=[45.0, 0.0, 10.0],
            length_y=1.6,
            length_z=1.6,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[-1.0, -1.0, 1.8],
            rpy=[-0.0, -0.0, 0.0],
            length_y=1.8,
            length_z=1.8,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[9.0, 1.3, 1.5],
            rpy=[-30.0, 10.0, 20.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[18.0, -1.0, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=2,
            length_z=2,
            gate=True,
        ),
    ]


def _define_obstacles() -> Tuple[List[urdfpy.Link], List[List[float]]]:
    obstacle_links = []
    obstacle_origins = []

    # cylinders
    obstacle_links.append(cylinder_link("obstacle_0", 0.3, 2.0))
    obstacle_origins.append([-14.0, -1.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_1", 0.3, 3.0))
    obstacle_origins.append([-13.0, 0.0, 1.5, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_2", 0.3, 1.5))
    obstacle_origins.append([-15.0, 0.5, 1.0, 0.0, 0.0, 0.0])

    # cuboids
    obstacle_links.append(cuboid_link("obstacle_3", [1.0, 1.0, 1.0]))
    obstacle_origins.append([-6.0, 1.25, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_4", [0.3, 2, 3.0]))
    obstacle_origins.append([-4.0, 1.5, 1.5, radians(45.0), 0.0, radians(15.0)])

    obstacle_links.append(cuboid_link("obstacle_5", [0.3, 2, 2.0]))
    obstacle_origins.append([-6.0, -1.5, 1.5, 0.0, 0.0, radians(-15.0)])

    obstacle_links.append(cuboid_link("obstacle_6", [0.6, 1.2, 0.9]))
    obstacle_origins.append([-5.0, -0.4, 2.5, 0.0, 20.0, radians(-15.0)])

    # cuboid wireframes
    obstacle_links.append(cuboid_wireframe_link("obstacle_7", [1.3, 1.3, 1.3], 0.5))
    obstacle_origins.append([4.5, -1, -0.5, 0.0, 0.0, radians(-30.0)])

    obstacle_links.append(cuboid_wireframe_link("obstacle_8", [2.5, 2.5, 2.5], 0.33))
    obstacle_origins.append([4, 1.0, 2.0, 45.0, 0.0, radians(45.0)])

    # spheres
    obstacle_links.append(sphere_link("obstacle_9", 1.0))
    obstacle_origins.append([13.5, 0.0, 2, 0.0, 0.0, 0.0])

    obstacle_links.append(sphere_link("obstacle_10", 0.6))
    obstacle_origins.append([13, 1.5, 1, 0.0, 0.0, 0.0])

    obstacle_links.append(sphere_link("obstacle_11", 0.8))
    obstacle_origins.append([14, 2.75, 2, 0.0, 0.0, 0.0])

    return obstacle_links, obstacle_origins
