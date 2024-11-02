from dataclasses import dataclass

from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase


@dataclass
class CollectionCapsuleOptions(CollectionBaseOptions):
    """
    Params definition: [``radius``, ``length``]
    """

    params_min = [0.1, 0.2]
    params_max = [1.0, 1.0]


class CollectionCapsule(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        radius, length = self.params_rand[blueprint_id]
        asset = self.gym.create_capsule(
            self.sim, radius, length, self.options.asset_options
        )
        return asset
