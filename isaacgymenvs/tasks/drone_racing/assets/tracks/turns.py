from dataclasses import dataclass
from typing import List, Tuple

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils import TrackOptions
from ..utils.track_utils import create_track_asset
from ...waypoint import Waypoint


@dataclass
class TrackTurnsOptions:
    # file name
    file_name: str = "track_turns"

    # common options for racing tracks
    track_options: TrackOptions = TrackOptions()

    # options for importing into Isaac Gym
    asset_options: AssetOptions = AssetOptions()


def create_track_turns(
    gym: Gym,
    sim: Sim,
    options: TrackTurnsOptions,
) -> Tuple[Asset, List[Waypoint]]:

    waypoints: List[Waypoint] = [
        Waypoint(
            index=0,
            xyz=[-10.0, -15.0, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=1,
            xyz=[-0.0, -15.0, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=2,
            xyz=[10, -15.0, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=3,
            xyz=[15, -7.5, 1.5],
            rpy=[0.0, 0.0, 90.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=4,
            xyz=[10, 0.0, 1.5],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=5,
            xyz=[0, 0.0, 1.5],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=6,
            xyz=[-10, 0.0, 1.5],
            rpy=[0.0, 0.0, 180.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=7,
            xyz=[-15, 7.5, 1.5],
            rpy=[0.0, 0.0, 90.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=8,
            xyz=[-10, 15, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=9,
            xyz=[-0, 15, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
        Waypoint(
            index=10,
            xyz=[10, 15, 1.5],
            rpy=[0.0, 0.0, 0.0],
            length_y=1.7,
            length_z=1.7,
            gate=True,
        ),
    ]
    asset = create_track_asset(
        options.file_name,
        options.track_options,
        waypoints,
        [],
        [],
        [],
        options.asset_options,
        gym,
        sim,
    )
    return asset, waypoints
