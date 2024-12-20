from dataclasses import dataclass

from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase
from ..geometries import GeomCylinderOptions, create_geom_cylinder


@dataclass
class CollectionCylinderOptions(CollectionBaseOptions):
    """
    Params: [``radius``, ``length``]
    """

    params_min = [0.1, 0.2]
    params_max = [1.0, 2.0]


class CollectionCylinder(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        radius, length = self.params_rand[blueprint_id]
        opts = GeomCylinderOptions()
        opts.radius = radius
        opts.length = length
        opts.asset_options = self.options.asset_options
        asset = create_geom_cylinder(self.gym, self.sim, opts)
        return asset
