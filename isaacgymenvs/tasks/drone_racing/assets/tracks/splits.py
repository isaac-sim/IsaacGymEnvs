from dataclasses import dataclass
from typing import List, Tuple

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ...waypoint import Waypoint


@dataclass
class TrackSplitsOptions:
    # file name
    file_name: str = "track_splits"

    # common options for racing tracks
    track_options: TrackOptions = TrackOptions()

    # options for importing into Isaac Gym
    asset_options: AssetOptions = AssetOptions()


def create_track_splits(
    gym: Gym,
    sim: Sim,
    options: TrackSplitsOptions,
) -> Tuple[Asset, List[Waypoint]]:
    """
    Create the Split-S track.

    References
        - https://arxiv.org/abs/2403.12203

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
    obstacle_links = []
    obstacle_origins = []
    obstacle_flags = []

    return waypoints, obstacle_links, obstacle_origins, obstacle_flags


def _define_waypoints() -> List[Waypoint]:
    waypoints = [
        Waypoint(
            index=0,
            xyz=[-5.0, 4.75, 1.0],
            rpy=[0.0, 0.0, -90.0],
            length_y=1.0,
            length_z=1.0,
            gate=False,
        ),
        Waypoint(
            index=1,
            xyz=[-0.5, -1.0, 3.25],
            rpy=[0.0, 0.0, -18.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[9.6, 6.25, 1.1],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[9.5, -3.8, 1.1],
            rpy=[0.0, 0.0, 226.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[-4.5, -5.1, 3.25],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=5,
            xyz=[-4.5, -5.1, 1.2],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=6,
            xyz=[4.9, -0.5, 1.1],
            rpy=[0.0, 0.0, 79.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[-2.0, 6.6, 1.1],
            rpy=[0.0, 0.0, 208.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=8,
            xyz=[-0.5, -1.0, 3.25],
            rpy=[0.0, 0.0, -18.0],
            length_y=1.7,
            length_z=1.7,
            gate=False,
        ),
    ]

    return waypoints
