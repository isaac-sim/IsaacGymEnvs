import os
from dataclasses import dataclass

from isaacgym.gymapi import Asset
from .base import CollectionBaseOptions, CollectionBase


@dataclass
class CollectionTreeOptions(CollectionBaseOptions):
    params_min = None
    params_max = None


class CollectionTree(CollectionBase):
    def create_asset(self, blueprint_id: int) -> Asset:
        asset_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../../../assets/urdf/aerial_gym_trees",
        )
        asset = self.gym.load_asset(
            self.sim,
            asset_dir,
            "tree_" + str(blueprint_id) + ".urdf",
            self.options.asset_options,
        )
        return asset
