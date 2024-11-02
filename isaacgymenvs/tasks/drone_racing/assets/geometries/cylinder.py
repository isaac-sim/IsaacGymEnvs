from dataclasses import dataclass

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils.urdf_utils import cylinder_link, export_urdf


@dataclass
class GeomCylinderOptions:
    file_name: str = "geom_cylinder"
    radius: float = 0.5
    length: float = 1.0
    asset_options: AssetOptions = AssetOptions()


def create_geom_cylinder(gym: Gym, sim: Sim, options: GeomCylinderOptions) -> Asset:
    link = cylinder_link("base", options.radius, options.length)
    urdf = urdfpy.URDF(options.file_name, [link])
    file_dir, file_name_ext = export_urdf(urdf)
    asset = gym.load_asset(sim, file_dir, file_name_ext, options.asset_options)
    return asset
