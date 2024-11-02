from dataclasses import dataclass
from math import sin, cos, radians
from typing import List, Tuple

import numpy as np
import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ..utils.urdf_utils import cuboid_link, cylinder_link
from ...waypoint import Waypoint


@dataclass
class TrackMultiStoryOptions:
    # file name
    file_name: str = "track_multistory"

    # common options for racing tracks
    track_options: TrackOptions = TrackOptions()

    # options for importing into Isaac Gym
    asset_options: AssetOptions = AssetOptions()

    # number of random cylinders
    num_cylinders: int = 40

    # radius of cylinders
    radius_cylinders: float = 0.25

    # no obstacle exists within waypoint clearance
    waypoint_clearance: float = 2.0

    # minimum distance between two cylinders
    min_dist_cylinders: float = 2.0

    # maximum attempts for sampling a new cylinder
    max_num_attempts: int = 100


def create_track_multistory(
    gym: Gym, sim: Sim, options: TrackMultiStoryOptions
) -> Tuple[Asset, List[Waypoint]]:
    """
    Create the Multi-story Arena track with random cylinder obstacles.

    References
        - https://github.com/uzh-rpg/sb_min_time_quadrotor_planning

    Args:
        gym: returned by ``acquire_gym``.
        sim: simulation handle.
        options: options for the asset, and importing.

    Returns:
        - An asset object as the return of calling ``load_asset``.
        - A list of ``Waypoint`` instances.
    """

    waypoints, obstacle_links, obstacle_origins, obstacle_flags = _define_track(options)
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


def _define_track(options: TrackMultiStoryOptions) -> Tuple[
    List[Waypoint],
    List[urdfpy.Link],
    List[List[float]],
    List[bool],
]:
    waypoints = _define_waypoints()
    obstacle_links, obstacle_origins = _define_obstacles()
    obstacle_flags = [False] * len(obstacle_links)
    _add_random_obstacles(waypoints, obstacle_links, obstacle_origins, options)
    obstacle_flags += [True] * (len(obstacle_links) - len(obstacle_flags))

    return waypoints, obstacle_links, obstacle_origins, obstacle_flags


def _define_waypoints() -> List[Waypoint]:
    waypoints = [
        Waypoint(
            index=0,
            xyz=[-7.0, 2.5, 1.0],
            rpy=[0.0, 0.0, -90.0],
            length_y=1.0,
            length_z=1.0,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[7.17 * cos(radians(202)), 7.17 * sin(radians(202)), 1.2],
            rpy=[0.0, 0.0, -69.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[0.07, -2.75, 2.8],
            rpy=[0.0, -90.0, 90.0],
            length_y=2.0,
            length_z=1.68,
            gate=False,
        ),
        Waypoint(
            index=3,
            xyz=[1.4, -0.6, 4.1],
            rpy=[0.0, 0.0, 90.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[-1.25, 2.95, 2.8],
            rpy=[0.0, 90.0, 90.0],
            length_y=2.0,
            length_z=1.68,
            gate=False,
        ),
        Waypoint(
            index=5,
            xyz=[10.95 * cos(radians(127.7)), 10.95 * sin(radians(127.7)), 1.2],
            rpy=[0.0, 0.0, 70.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=6,
            xyz=[9.88 * cos(radians(68.5)), 9.88 * sin(radians(68.5)), 1.2],
            rpy=[0.0, 0.0, -37.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[6.4, 1.55, 2.8],
            rpy=[0.0, -90.0, -90.0],
            length_y=2.0,
            length_z=3.36,
            gate=False,
        ),
        Waypoint(
            index=8,
            xyz=[6.0, 6.67 * sin(radians(-21)), 4.1],
            rpy=[0.0, 0.0, -90.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=9,
            xyz=[6.4, -4.7, 2.8],
            rpy=[0.0, 90.0, 90.0],
            length_y=2.0,
            length_z=1.68,
            gate=False,
        ),
        Waypoint(
            index=10,
            xyz=[6.67 * cos(radians(-21)), 6.67 * sin(radians(-21)), 1.2],
            rpy=[0.0, 0.0, 90.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=11,
            xyz=[4.28 * cos(radians(81)), 4.28 * sin(radians(81)), 1.2],
            rpy=[0.0, 0.0, 162.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
    ]

    return waypoints


def _define_obstacles() -> Tuple[List[urdfpy.Link], List[List[float]]]:
    obstacle_links = []
    obstacle_origins = []

    obstacle_links.append(cuboid_link("obstacle_0", [4.2, 20.0, 0.15]))
    obstacle_origins.append([3.25, 3.0, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_1", [2.0, 20.0, 0.15]))
    obstacle_origins.append([8.5, 3.0, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_2", [2.2, 1.4, 0.15]))
    obstacle_origins.append([6.4, -6.3, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_3", [2.2, 3.6, 0.15]))
    obstacle_origins.append([6.4, -2.0, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_4", [2.2, 9.75, 0.15]))
    obstacle_origins.append([6.4, 8.125, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_5", [2.2, 9.75, 0.15]))
    obstacle_origins.append([6.4, 8.125, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_6", [8.1, 20.0, 0.15]))
    obstacle_origins.append([-6.4, 3.0, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_7", [4.0, 9.15, 0.15]))
    obstacle_origins.append([-0.8, 8.425, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_8", [4.0, 3.3, 0.15]))
    obstacle_origins.append([-0.8, -5.35, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_9", [4.0, 3.9, 0.15]))
    obstacle_origins.append([-0.8, 0.1, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_10", [2.0, 2.0, 0.15]))
    obstacle_origins.append([0.8, 2.95, 2.8, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_11", [2.0, 2.0, 0.15]))
    obstacle_origins.append([-2.0, -2.75, 2.8, 0.0, 0.0, 0.0])

    return obstacle_links, obstacle_origins


def _add_random_obstacles(
    waypoints: List[Waypoint],
    obstacle_links: List[urdfpy.Link],
    obstacle_origins: List[List[float]],
    options: TrackMultiStoryOptions,
):
    rng = np.random.default_rng()

    random_obstacle_id = 0
    random_obstacle_links = []
    random_obstacle_origins = []
    num_attempts = 0
    while len(random_obstacle_links) < options.num_cylinders:
        if num_attempts > options.max_num_attempts:
            break
        random_xy = rng.uniform(-1, 1, 2) * 9.5 + np.array([-1, 3])
        random_xyz = np.array([random_xy[0], random_xy[1], 1.0])
        # check distance to waypoint center
        xyz_is_valid = True
        for waypoint in waypoints:
            distance = np.linalg.norm(np.array(waypoint.xyz) - random_xyz)
            if distance < options.waypoint_clearance:
                xyz_is_valid = False
                break
        # check distance to other cylinders
        for random_obstacle_origin in random_obstacle_origins:
            distance = np.linalg.norm(np.array(random_obstacle_origin[:3]) - random_xyz)
            if distance < options.min_dist_cylinders:
                xyz_is_valid = False
                break
        if not xyz_is_valid:
            num_attempts += 1
            continue
        else:
            random_obstacle_links.append(
                cylinder_link(
                    "random_obstacle_" + str(random_obstacle_id),
                    options.radius_cylinders,
                    2.0,
                )
            )
            random_obstacle_origins.append(
                [random_xy[0], random_xy[1], 1.0, 0.0, 0.0, 0.0]
            )
            random_obstacle_id += 1
            num_attempts = 0

    obstacle_links += random_obstacle_links
    obstacle_origins += random_obstacle_origins
