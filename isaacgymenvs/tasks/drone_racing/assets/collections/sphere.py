from dataclasses import dataclass

from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase


@dataclass
class CollectionSphereOptions(CollectionBaseOptions):
    """
    Params: [``radius``]
    """

    params_min = [0.1]
    params_max = [1.0]


class CollectionSphere(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        r = self.params_rand[blueprint_id][0]
        asset = self.gym.create_sphere(self.sim, r, self.options.asset_options)
        return asset
