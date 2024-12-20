import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset, get_line_segment
from ..utils.urdf_utils import (
    cuboid_wireframe_link,
    cuboid_link,
    cylinder_link,
    random_geometries_link,
)
from ...waypoint import Waypoint


@dataclass
class TrackRmuaOptions:
    # file name
    file_name: str = "track_rmua"

    # common options for racing tracks
    track_options: TrackOptions = TrackOptions()

    # options for importing into Isaac Gym
    asset_options: AssetOptions = AssetOptions()

    # flag to enable waypoint randomization
    enable_waypoint_randomization: bool = False

    # flag to enable additional random obstacles
    enable_additional_obstacles: bool = False


def create_track_rmua(
    gym: Gym,
    sim: Sim,
    options: TrackRmuaOptions,
) -> Tuple[Asset, List[Waypoint]]:
    """
    Create the RMUA 2023 racing track with gate pose randomization and additional obstacles.

    References
        - https://www.robomaster.com/zh-CN/resource/pages/announcement/1644

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


def _define_track(
    options: TrackRmuaOptions,
) -> Tuple[List[Waypoint], List[urdfpy.Link], List[List[float]], List[bool]]:
    waypoints = _define_waypoints()
    obstacle_links, obstacle_origins = _define_obstacles()
    obstacle_flags = [False] * len(obstacle_links)

    if options.enable_waypoint_randomization:
        _modify_waypoints(waypoints)

    if options.enable_additional_obstacles:
        _add_random_obstacles(waypoints, obstacle_links, obstacle_origins)
        obstacle_flags += [True] * (len(obstacle_links) - len(obstacle_flags))

    return waypoints, obstacle_links, obstacle_origins, obstacle_flags


def _define_waypoints() -> List[Waypoint]:
    waypoints = [
        Waypoint(
            index=0,
            xyz=[0.0, 0.0, 1.0],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.0,
            length_z=1.0,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[-4.0, -0.5, 1.5],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[-10.0, -0.25, 1.0],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[-12.25, 5.25, 1.1],
            rpy=[0.0, 0.0, 45.0],
            length_y=1.5,
            length_z=1.8,
            gate=False,
        ),
        Waypoint(
            index=4,
            xyz=[-7.5, 4.25, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=5,
            xyz=[-3.5, 5.0, 0.9],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.0,
            length_z=1.6,
            gate=False,
        ),
        Waypoint(
            index=6,
            xyz=[1.0, 4.25, 0.65],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[6.5, 5.6, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=8,
            xyz=[6.5, 0.0, 6.5],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=9,
            xyz=[6.5, 5.5, 8.0],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=10,
            xyz=[8.0, 0.2, 1.0],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.0,
            length_z=1.0,
            gate=True,
        ),
        Waypoint(
            index=11,
            xyz=[2.0, 0.2, 0.75],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.2,
            length_z=1.2,
            gate=False,
        ),
    ]

    return waypoints


def _define_obstacles() -> Tuple[List[urdfpy.Link], List[List[float]]]:
    obstacle_links = []
    obstacle_origins = []

    obstacle_links.append(cuboid_link("obstacle_0", [0.6, 1.8, 2.1]))
    obstacle_origins.append([-11.35, 3.75, 1.05, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_1", [0.6, 1.2, 2.1]))
    obstacle_origins.append([-13.2, 6.45, 1.05, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_2", [1.0, 1.5, 15.0]))
    obstacle_origins.append([6.5, 4.0, 6.0, math.radians(45.0), 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_3", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 4.0, 0.325, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_4", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 6.0, 0.325, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_5", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 6.25, 0.975, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_6", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.25, 4.0, 0.975, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_7", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.75, 6.0, 1.625, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_8", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 4.0, 1.625, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_9", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 4.5, 2.275, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_10", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 5.15, 2.275, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_wireframe_link("obstacle_11", [0.6, 0.6, 0.6], 0.05))
    obstacle_origins.append([-3.5, 5.8, 2.275, 0.0, 0.0, 0.0])

    obstacle_links.append(cuboid_link("obstacle_12", [0.6, 0.6, 2.0]))
    obstacle_origins.append([4.0, 5.0, 1.0, 0.0, 0.0, math.radians(45)])

    obstacle_links.append(cylinder_link("obstacle_13", 0.1, 2.0))
    obstacle_origins.append([6.0, -0.7, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_14", 0.1, 2.0))
    obstacle_origins.append([6.0, 1.1, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_15", 0.1, 2.0))
    obstacle_origins.append([2.0, -0.7, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_16", 0.1, 2.0))
    obstacle_origins.append([2.0, 1.1, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_17", 0.1, 2.0))
    obstacle_origins.append([4.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_18", 0.1, 1.0))
    obstacle_origins.append([5.6, -0.8, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_19", 0.1, 1.0))
    obstacle_origins.append([6.3, -1.05, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_20", 0.1, 1.0))
    obstacle_origins.append([6.3, -0.28, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_21", 0.1, 1.0))
    obstacle_origins.append([5.6, 0.8, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_22", 0.1, 1.0))
    obstacle_origins.append([6.3, 0.67, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_23", 0.1, 1.0))
    obstacle_origins.append([6.3, 1.4, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_24", 0.1, 1.0))
    obstacle_origins.append([3.5, 0.0, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_25", 0.1, 1.0))
    obstacle_origins.append([4.2, -0.3, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_26", 0.1, 1.0))
    obstacle_origins.append([4.2, 0.3, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_27", 0.1, 1.0))
    obstacle_origins.append([1.6, -0.8, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_28", 0.1, 1.0))
    obstacle_origins.append([2.3, -1.05, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_29", 0.1, 1.0))
    obstacle_origins.append([2.3, -0.28, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_30", 0.1, 1.0))
    obstacle_origins.append([1.6, 0.8, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_31", 0.1, 1.0))
    obstacle_origins.append([2.3, 0.67, 2.0, 0.0, 0.0, 0.0])

    obstacle_links.append(cylinder_link("obstacle_32", 0.1, 1.0))
    obstacle_origins.append([2.3, 1.4, 2.0, 0.0, 0.0, 0.0])

    return obstacle_links, obstacle_origins


def _modify_waypoints(waypoints: List[Waypoint]):
    num_waypoints = len(waypoints)
    scale = torch.zeros(num_waypoints, 6)
    offset = torch.zeros(num_waypoints, 6)
    rand_gen = torch.rand(num_waypoints, 6)
    rand = rand_gen * 2 - 1  # [-1, 1)

    # waypoint 1
    scale[1] = torch.tensor([1.0, 1.0, 0.5, 10.0, 10.0, 10.0])

    # waypoint 2
    scale[2] = torch.tensor([1.0, 1.0, 0.5, 20.0, 20.0, 20.0])
    offset[2] = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

    # waypoint 4
    scale[4] = torch.tensor([0.0, 1.0, 0.5, 20.0, 10.0, 20.0])
    offset[4] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # waypoint 6
    scale[6] = torch.tensor([0.5, 0.5, 0.5, 20.0, 10.0, 20.0])
    offset[6] = torch.tensor([-0.5, 0.5, 0.5, 0.0, 0.0, 0.0])

    # waypoint 7
    scale[7] = torch.tensor([0.5, 1.0, 0.0, 20.0, 20.0, 20.0])
    offset[7] = torch.tensor([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])

    # waypoint 8
    scale[8] = torch.tensor([1.0, 1.0, 1.0, 20.0, 20.0, 20.0])

    # waypoint 9
    scale[9] = torch.tensor([1.0, 1.0, 1.0, 20.0, 20.0, 20.0])

    # waypoint 10
    scale[10] = torch.tensor([1.0, 1.0, 0.5, 20.0, 20.0, 20.0])
    offset[10] = torch.tensor([1.0, 0.0, 0.5, 0.0, 0.0, 0.0])

    mod = rand * scale + offset
    for i in range(num_waypoints):
        waypoint = waypoints[i]
        waypoint.xyz = (torch.tensor(waypoint.xyz) + mod[i, :3]).tolist()
        waypoint.rpy = (torch.tensor(waypoint.rpy) + mod[i, 3:]).tolist()


def _add_random_obstacles(
    waypoints: List[Waypoint],
    obstacle_links: List[urdfpy.Link],
    obstacle_origins: List[List[float]],
):
    # between waypoint 1, 2
    xyz_rpy, length = get_line_segment(waypoints[1].xyz, waypoints[2].xyz)
    obstacle_links.append(
        random_geometries_link(
            "random_obstacle_0",
            6,
            [max(2.0, length - 3), 3, 3],
            [0.0, 0.0, 0.0],
            0.25,
            0.75,
        )
    )
    obstacle_origins.append(xyz_rpy)

    # between waypoint 2, 3
    xyz_rpy, length = get_line_segment(waypoints[2].xyz, waypoints[3].xyz)
    obstacle_links.append(
        random_geometries_link(
            "random_obstacle_1",
            10,
            [max(2.0, length - 2), 4, 3],
            [0.0, 2.0, 0.0],
            0.25,
            0.75,
        )
    )
    obstacle_origins.append(xyz_rpy)

    # between waypoint 3, 4
    xyz_rpy, length = get_line_segment(waypoints[3].xyz, waypoints[4].xyz)
    obstacle_links.append(
        random_geometries_link(
            "random_obstacle_2",
            5,
            [max(2.0, length - 3), 2, 2],
            [0.0, 1.0, 0.0],
            0.25,
            0.5,
        )
    )
    obstacle_origins.append(xyz_rpy)

    # between waypoint 6, 7
    xyz_rpy, length = get_line_segment(waypoints[6].xyz, waypoints[7].xyz)
    obstacle_links.append(
        random_geometries_link(
            "random_obstacle_3",
            5,
            [max(1.0, length - 4), 2, 2],
            [0.0, 0.0, 0.0],
            0.1,
            1.0,
        )
    )
    obstacle_origins.append(xyz_rpy)

    # between waypoint 7, 8
    xyz_rpy, length = get_line_segment(waypoints[7].xyz, waypoints[8].xyz)
    obstacle_links.append(
        random_geometries_link(
            "random_obstacle_4",
            20,
            [max(2.0, length - 2), 5, 3],
            [0.0, 3.0, 0.0],
            0.25,
            0.75,
        )
    )
    obstacle_origins.append(xyz_rpy)

    # between waypoint 8, 9
    xyz_rpy, length = get_line_segment(waypoints[8].xyz, waypoints[9].xyz)
    obstacle_links.append(
        random_geometries_link(
            "random_obstacle_5",
            20,
            [max(2.0, length - 2), 6, 6],
            [0.0, 3.0, 0.0],
            0.25,
            0.75,
        )
    )
    obstacle_origins.append(xyz_rpy)
