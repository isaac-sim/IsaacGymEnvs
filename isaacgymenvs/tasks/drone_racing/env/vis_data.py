from typing import List

import torch

from isaacgym import gymutil
from isaacgym.gymapi import Gym, Vec3, Env, Viewer, Transform, Quat


class OrbitVisData:

    def __init__(
        self,
        position: torch.Tensor,
        r_min: torch.Tensor,
        r_mean: torch.Tensor,
        r_max: torch.Tensor,
    ):
        """
        Args:
            position: tensor in (num_envs, num_orbits, 3).
            r_min: tensor in (num_envs, num_orbits).
            r_mean: tensor in (num_envs, num_orbits).
            r_max: tensor in (num_envs, num_orbits).
        """
        self.position = position
        self.r_min = r_min
        self.r_mean = r_mean
        self.r_max = r_max

    @property
    def num_envs(self):
        return int(self.position.shape[0])

    @property
    def num_orbits(self):
        return int(self.position.shape[1])

    def visualize(
        self,
        gym: Gym,
        envs: List[Env],
        viewer: Viewer,
        draw_min: bool = True,
        draw_mean: bool = True,
        draw_max: bool = True,
        min_color: List[float] = None,
        mean_color: List[float] = None,
        max_color: List[float] = None,
    ):
        num_envs = len(envs)
        assert num_envs == self.num_envs

        if min_color is None:
            min_color = (1.0, 1.0, 1.0)
        if mean_color is None:
            mean_color = (0.0, 0.0, 1.0)
        if max_color is None:
            max_color = (0.6, 0.2, 0.7)

        tf = Transform()
        for i in range(num_envs):
            for j in range(self.num_orbits):
                x, y, z = self.position[i, j].tolist()
                tf.p = Vec3(x, y, z)
                if draw_min:
                    sphere = gymutil.WireframeSphereGeometry(
                        radius=self.r_min[i, j], color=min_color
                    )
                    gymutil.draw_lines(sphere, gym, viewer, envs[i], tf)
                if draw_mean:
                    sphere = gymutil.WireframeSphereGeometry(
                        radius=self.r_mean[i, j], color=mean_color
                    )
                    gymutil.draw_lines(sphere, gym, viewer, envs[i], tf)
                if draw_max:
                    sphere = gymutil.WireframeSphereGeometry(
                        radius=self.r_max[i, j], color=max_color
                    )
                    gymutil.draw_lines(sphere, gym, viewer, envs[i], tf)


class WallRegionVisData:

    def __init__(
        self, position: torch.Tensor, quaternion: torch.Tensor, dim: torch.Tensor
    ):
        """
        Args:
            position: tensor in (num_envs, num_regions, 3).
            quaternion: tensor in (num_envs, num_regions, 4).
            dim: tensor in (num_envs, num_regions).
        """
        self.position = position
        self.quaternion = quaternion
        self.dim = dim

    @property
    def num_envs(self):
        return int(self.position.shape[0])

    @property
    def num_regions(self):
        return int(self.position.shape[1])

    def visualize(
        self,
        gym: Gym,
        envs: List[Env],
        viewer: Viewer,
        color: List[float] = None,
    ):
        num_envs = len(envs)
        assert num_envs == self.num_envs

        if color is None:
            color = (1.0, 1.0, 0.0)

        tf = Transform()
        for i in range(num_envs):
            for j in range(self.num_regions):
                x, y, z = self.position[i, j]
                qx, qy, qz, qw = self.quaternion[i, j]
                tf.p = Vec3(x, y, z)
                tf.r = Quat(qx, qy, qz, qw)
                dim = self.dim[i, j]
                square = gymutil.WireframeBoxGeometry(dim, dim, dim, color=color)
                gymutil.draw_lines(square, gym, viewer, envs[i], tf)
