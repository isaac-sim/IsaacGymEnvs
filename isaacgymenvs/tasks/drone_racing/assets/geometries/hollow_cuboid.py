from dataclasses import dataclass, field
from typing import List

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils.urdf_utils import hollow_cuboid_link, export_urdf


@dataclass
class GeomHollowCuboidOptions:
    file_name: str = "geom_hollow_cuboid"
    length_x: float = 0.15
    inner_length_y: float = 1.0
    outer_length_y: float = 1.3
    inner_length_z: float = 1.0
    outer_length_z: float = 1.3
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    asset_options: AssetOptions = AssetOptions()


def create_geom_hollow_cuboid(
    gym: Gym, sim: Sim, options: GeomHollowCuboidOptions
) -> Asset:
    link = hollow_cuboid_link(
        "base",
        options.length_x,
        options.inner_length_y,
        options.outer_length_y,
        options.inner_length_z,
        options.outer_length_z,
        options.color,
    )
    urdf = urdfpy.URDF(options.file_name, [link])
    file_dir, file_name_ext = export_urdf(urdf)
    asset = gym.load_asset(sim, file_dir, file_name_ext, options.asset_options)
    return asset
