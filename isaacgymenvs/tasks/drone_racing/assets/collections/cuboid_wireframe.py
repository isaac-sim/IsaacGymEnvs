from dataclasses import dataclass

from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase
from ..geometries import GeomCuboidWireframeOptions, create_geom_cuboid_wireframe


@dataclass
class CollectionCuboidWireframeOptions(CollectionBaseOptions):
    """
    Params: [``size_x``, ``size_y``, ``size_z``, ``weight``]
    """

    params_min = [0.4, 0.4, 0.4, 0.1]
    params_max = [2.0, 2.0, 2.0, 0.4]


class CollectionCuboidWireframe(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        x, y, z, w = self.params_rand[blueprint_id]
        min_xyz = min(x, y, z)
        if w > min_xyz / 2:
            w = min_xyz / 2
        opts = GeomCuboidWireframeOptions()
        opts.size = [x, y, z]
        opts.weight = w
        opts.asset_options = self.options.asset_options
        asset = create_geom_cuboid_wireframe(self.gym, self.sim, opts)
        return asset
