import abc
from dataclasses import dataclass
from typing import List, Optional

import torch
from tqdm import tqdm

from isaacgym.gymapi import Gym, Sim, Asset, AssetOptions


@dataclass
class CollectionBaseOptions:
    # number of envs
    num_envs: int = 64

    # number of assets per env
    num_assets: int = 10

    # number of total random blueprints
    num_blueprints: int = 100

    # asset options
    asset_options: AssetOptions = AssetOptions()

    # if true, the process of loading assets won't be printed
    disable_tqdm: bool = False

    # minimum value of params
    params_min: Optional[List[float]] = None

    # maximum value of params
    params_max: Optional[List[float]] = None


class CollectionBase:
    def __init__(self, gym: Gym, sim: Sim, options: CollectionBaseOptions):
        self.gym = gym
        self.sim = sim
        self.options: CollectionBaseOptions = options

        self.params_rand: Optional[List[List[float]]] = None
        self.ids_rand: Optional[List[List[int]]] = None
        self.generate_rand()

        self.assets: Optional[List[List[Asset]]] = None
        self.load_assets()

    def generate_rand(self):
        # random params for blueprints
        if self.options.params_min is not None and self.options.params_max is not None:
            assert len(self.options.params_min) == len(self.options.params_max)
            num_params = len(self.options.params_min)

            p_min = torch.tensor(self.options.params_min)
            p_min.clamp_(min=0.01)
            p_max = torch.tensor(self.options.params_max)
            p_max.clamp_(min=p_min)
            p_range = p_max - p_min

            p_rand = torch.rand(self.options.num_blueprints, num_params)
            p_rand = p_rand * p_range + p_min

            self.params_rand = p_rand.tolist()

        else:
            self.params_rand = []

        # random ids for selecting blueprints
        if self.options.num_assets > 0:
            self.ids_rand = torch.randint(
                0,
                self.options.num_blueprints,
                (self.options.num_envs, self.options.num_assets),
            ).tolist()
        else:
            self.ids_rand = []

    def load_assets(self):
        # create blueprints
        blueprints = []
        for i in tqdm(
            range(self.options.num_blueprints), disable=self.options.disable_tqdm
        ):
            blueprints.append(self.create_asset(i))

        # fill the assets list
        self.assets = []
        for i in range(self.options.num_envs):
            env_assets = []
            for j in range(self.options.num_assets):
                index = self.ids_rand[i][j]
                env_assets.append(blueprints[index])
            self.assets.append(env_assets)

    @abc.abstractmethod
    def create_asset(self, blueprint_id: int) -> Asset:
        """
        This method loads assets into the list ``assets``.
        """

        raise NotImplementedError
