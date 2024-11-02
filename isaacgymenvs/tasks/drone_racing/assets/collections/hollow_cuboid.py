from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase
from ..geometries import GeomHollowCuboidOptions, create_geom_hollow_cuboid


class CollectionHollowCuboidOptions(CollectionBaseOptions):
    """
    Params: [``length_x``, ``inner_length_y``, ``inner_length_z``, ``diff_length_y``, ``diff_length_z``]
    """

    params_min = [0.05, 0.5, 0.5, 0.2, 0.2]
    params_max = [0.2, 1.5, 1.5, 0.8, 0.8]


class CollectionHollowCuboid(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        x, in_y, in_z, d_y, d_z = self.params_rand[blueprint_id]
        opts = GeomHollowCuboidOptions()
        opts.length_x = x
        opts.inner_length_y = in_y
        opts.inner_length_z = in_z
        opts.outer_length_y = in_y + d_y
        opts.outer_length_z = in_z + d_z
        opts.asset_options = self.options.asset_options
        asset = create_geom_hollow_cuboid(self.gym, self.sim, opts)
        return asset
