from dataclasses import dataclass
from math import radians
from typing import List, Tuple

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ..utils.urdf_utils import cuboid_wireframe_link, sphere_link, cuboid_link
from ...waypoint import Waypoint


@dataclass
class TrackSimpleStickOptions:
    file_name: str = "track_simple_stick"
    track_options: TrackOptions = TrackOptions()
    asset_options: AssetOptions = AssetOptions()
    add_obstacles: bool = True


def create_track_simple_stick(
    gym: Gym, sim: Sim, options: TrackSimpleStickOptions
) -> Tuple[Asset, List[Waypoint]]:

    waypoints = [
        Waypoint(
            index=0,
            xyz=[-18.0, -18.0, 1.5],
            rpy=[0.0, 0.0, 45.0],
            length_y=1.0,
            length_z=1.0,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[-10.0, -10.0, 1.4],
            rpy=[0.0, 0.0, 45.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[-2.0, -2.0, 1.5],
            rpy=[0.0, 0.0, 45.0],
            length_y=1.6,
            length_z=2.0,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[6, 6, 1.6],
            rpy=[0.0, 0.0, 45.0],
            length_y=1.8,
            length_z=1.8,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[14, 14, 1.5],
            rpy=[0.0, 0.0, 45.0],
            length_y=2.0,
            length_z=2.0,
            gate=True,
        ),
    ]

    obstacle_links = []
    obstacle_origins = []
    if options.add_obstacles:
        obstacle_links.append(
            cuboid_wireframe_link("obstacle_0", [2.3, 2.3, 2.3], 0.33)
        )
        obstacle_origins.append([-14, -14, 1.5, 0.0, 0.0, 0.0])

        obstacle_links.append(sphere_link("obstacle_1", 0.75))
        obstacle_origins.append([-6, -6, 1.5, 0.0, 0.0, 0.0])

        obstacle_links.append(cuboid_link("obstacle_2", [0.2, 2.0, 2]))
        obstacle_origins.append([2.0, 2, 1.5, 0.0, 0.0, radians(45)])

        obstacle_links.append(
            cuboid_wireframe_link("obstacle_3", [3.3, 3.3, 3.3], 0.33)
        )
        obstacle_origins.append([10.0, 10, 1.5, 0.0, 0.0, radians(45)])

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
