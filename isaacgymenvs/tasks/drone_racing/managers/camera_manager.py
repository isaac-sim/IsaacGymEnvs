from dataclasses import dataclass
from typing import List

import torch

from isaacgym import gymapi
from isaacgym.gymapi import Gym, Env


@dataclass
class RandCameraOptions:
    d_x_max: float = 0.01
    d_y_max: float = 0.0
    d_z_max: float = 0.01
    d_angle_max: float = 10.0


class CameraManager:

    def __init__(
        self,
        gym: Gym,
        cams: List[int],
        envs: List[Env],
        drones: List[int],
        init_cam_pos: List[float],
        init_cam_angle: float,
    ):

        self.gym: Gym = gym
        self.cams: List[int] = cams
        self.envs: List[Env] = envs
        self.drones: List[int] = drones
        self.num_envs = len(envs)
        self.init_cam_position = init_cam_pos
        self.init_cam_angle = init_cam_angle

    def randomize_camera_tf(self, options: RandCameraOptions) -> List[gymapi.Transform]:
        x = (
            torch.rand(self.num_envs) * 2 * options.d_x_max
            - options.d_x_max
            + self.init_cam_position[0]
        )
        y = (
            torch.rand(self.num_envs) * 2 * options.d_y_max
            - options.d_y_max
            + self.init_cam_position[1]
        )
        z = (
            torch.rand(self.num_envs) * 2 * options.d_z_max
            - options.d_z_max
            + self.init_cam_position[2]
        )
        angle = (
            torch.rand(self.num_envs) * 2 * options.d_angle_max
            - options.d_angle_max
            + self.init_cam_angle
        )

        cam_tf_list = []
        for i in range(self.num_envs):
            cam_tf = gymapi.Transform()
            cam_tf.p = gymapi.Vec3(x[i], y[i], z[i])
            cam_tf.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), -angle[i] * torch.pi / 180
            )
            if len(self.cams) > 0:
                self.gym.attach_camera_to_body(
                    self.cams[i],
                    self.envs[i],
                    self.drones[i],
                    cam_tf,
                    gymapi.FOLLOW_TRANSFORM,
                )
            cam_tf_list.append(cam_tf)

        return cam_tf_list
