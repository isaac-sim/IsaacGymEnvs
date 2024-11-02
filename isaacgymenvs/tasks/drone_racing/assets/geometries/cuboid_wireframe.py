from dataclasses import dataclass, field
from typing import List

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils.urdf_utils import cuboid_wireframe_link, export_urdf


@dataclass
class GeomCuboidWireframeOptions:
    file_name: str = "geom_cuboid_wireframe"
    size: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    weight: float = 0.1
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    asset_options: AssetOptions = AssetOptions()


def create_geom_cuboid_wireframe(
    gym: Gym, sim: Sim, options: GeomCuboidWireframeOptions
) -> Asset:
    link = cuboid_wireframe_link("base", options.size, options.weight, options.color)
    urdf = urdfpy.URDF(options.file_name, [link])
    file_dir, file_name_ext = export_urdf(urdf)
    asset = gym.load_asset(sim, file_dir, file_name_ext, options.asset_options)
    return asset
