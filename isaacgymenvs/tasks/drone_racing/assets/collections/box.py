from dataclasses import dataclass

from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase


@dataclass
class CollectionBoxOptions(CollectionBaseOptions):
    """
    Params: [``size_x``, ``size_y``, ``size_z``]
    """

    params_min = [0.2, 0.2, 0.2]
    params_max = [2.0, 2.0, 2.0]


class CollectionBox(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        x, y, z = self.params_rand[blueprint_id]
        asset = self.gym.create_box(self.sim, x, y, z, self.options.asset_options)
        return asset
