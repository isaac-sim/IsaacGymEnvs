from dataclasses import dataclass
from typing import List, Tuple

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ..utils.urdf_utils import (
    cuboid_link,
)
from ...waypoint import Waypoint


@dataclass
class TrackWallsOptions:
    # file name
    file_name: str = "track_walls"

    # common options for racing tracks
    track_options: TrackOptions = TrackOptions()

    # options for importing into Isaac Gym
    asset_options: AssetOptions = AssetOptions()


def create_track_walls(
    gym: Gym, sim: Sim, options: TrackWallsOptions
) -> Tuple[Asset, List[Waypoint]]:
    """
    Create the racing track with wall shaped obstacles.

    References
        - https://arxiv.org/abs/2203.15052

    Args:
        gym: returned by ``acquire_gym``.
        sim: simulation handle.
        options: options for the asset, and importing.

    Returns:
        - An asset object as the return of calling ``load_asset``.
        - A list of ``Waypoint`` instances.
    """

    waypoints, obstacle_links, obstacle_origins, obstacle_flags = _define_track()
    asset = create_track_asset(
        options.file_name,
        options.track_options,
        waypoints,
        obstacle_links,
        obstacle_origins,
        obstacle_flags,
        options.asset_options,
        gym,
        sim,
    )
    return asset, waypoints


def _define_track() -> Tuple[
    List[Waypoint],
    List[urdfpy.Link],
    List[List[float]],
    List[bool],
]:
    waypoints = _define_waypoints()
    obstacle_links, obstacle_origins = _define_obstacles()
    obstacle_flags = [False] * len(obstacle_links)

    return waypoints, obstacle_links, obstacle_origins, obstacle_flags


def _define_waypoints() -> List[Waypoint]:
    waypoints = [
        Waypoint(
            index=0,
            xyz=[-8.0, 3.0, 1.0],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.0,
            length_z=1.0,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[-4.0, 3.0, 1.0],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[0.0, 3.0, 1.0],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[3.0, 3.0, 1.0],
            rpy=[0.0, 0.0, -90.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[4.0, -1.0, 1.0],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=5,
            xyz=[0.0, -3.0, 1.0],
            rpy=[0.0, 0.0, -90.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=6,
            xyz=[0.0, -3.0, 3.0],
            rpy=[0.0, 0.0, 90.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[-4.0, -1.0, 3.0],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
    ]

    return waypoints


def _define_obstacles() -> Tuple[List[urdfpy.Link], List[List[float]]]:
    obstacle_links = []
    obstacle_origins = []

    obstacle_links.append(cuboid_link("obstacle_0", [4.0, 0.1, 2.0]))
    obstacle_origins.append([-4.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_1", [0.1, 2.0, 2.0]))
    obstacle_origins.append([-2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_2", [0.1, 2.0, 2.0]))
    obstacle_origins.append([2.0, 4.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_3", [2.0, 0.1, 2.0]))
    obstacle_origins.append([3.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_4", [2.0, 0.1, 2.0]))
    obstacle_origins.append([5.0, -2.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_5", [0.1, 2.0, 2.0]))
    obstacle_origins.append([1.0, -2.0, 1.0, 0.0, 0.0, 0.0])

    return obstacle_links, obstacle_origins
