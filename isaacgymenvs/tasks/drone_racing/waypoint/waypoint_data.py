from typing import List

import torch

from isaacgym import gymutil
from isaacgym.gymapi import Gym, Vec3, Env, Viewer, Transform, Quat
from isaacgymenvs.utils.torch_jit_utils import (
    quat_from_euler_xyz,
    quat_rotate_inverse,
    quaternion_to_matrix,
)
from .waypoint import Waypoint


class WaypointData:

    def __init__(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        gate_flag: torch.Tensor,
        gate_x_len_choice: torch.Tensor,
        gate_weight_choice: torch.Tensor,
        psi: torch.Tensor,
        theta: torch.Tensor,
        gamma: torch.Tensor,
        r: torch.Tensor,
    ):
        """
        Waypoint data and relative pose for multiple waypoints in parallel environments.

        Args:
            position: waypoint center position in world frame in (num_envs, num_waypoints, 3).
            quaternion: waypoint attitude in quaternion in (num_envs, num_waypoints, 4).
            width: region width (dim in waypoint body frame y) in (num_envs, num_waypoints).
            height: region height (dim in waypoint body frame z) in (num_envs, num_waypoints).
            gate_flag: {0.0, 1.0} indicating presence of a gate for waypoints in (num_envs, num_waypoints).
            gate_x_len_choice: {0.0, 1.0, ...} choice id of gate dim in waypoint body frame x
                in (num_envs, num_waypoints)
            gate_weight_choice: {0.0, 1.0, ...} choice id of gate weight (outer and inner dim diff of a hollow cube)
                in (num_envs, num_waypoints)
            psi: vector from this to next waypoint, this waypoint xy-plane in (num_envs, num_waypoints - 1)
            theta: angle between: vector from this to next waypoint, this waypoint x-axis
                in (num_envs, num_waypoints - 1)
            gamma: angle between: vector from this to next waypoint, next waypoint x-axis
                in (num_envs, num_waypoints - 1)
            r: distance between this and next waypoint in (num_envs, num_waypoints - 1)
        """

        self.position = position
        """(num_envs, num_waypoints, 3)"""

        self.quaternion = quaternion
        """(num_envs, num_waypoints, 4)"""

        self.width = width
        """(num_envs, num_waypoints)"""

        self.height = height
        """(num_envs, num_waypoints)"""

        self.gate_flag = gate_flag
        """(num_envs, num_waypoints)"""

        self.gate_x_len_choice = gate_x_len_choice
        """(num_envs, num_waypoints)"""

        self.gate_weight_choice = gate_weight_choice
        """(num_envs, num_waypoints)"""

        self.psi = psi
        """(num_envs, num_waypoints - 1)"""

        self.theta = theta
        """(num_envs, num_waypoints - 1)"""

        self.gamma = gamma
        """(num_envs, num_waypoints - 1)"""

        self.r = r
        """(num_envs, num_waypoints - 1)"""

    @classmethod
    def from_waypoint_list(
        cls,
        num_envs: int,
        waypoint_list: List[Waypoint],
        append_dummy: bool = False,
        append_dist: float = 10.0,
    ):
        waypoint_list = waypoint_list.copy()
        if append_dummy:
            last_wp = waypoint_list[-1]
            last_wp_compact = last_wp.compact_data()
            pos = torch.tensor(last_wp_compact[:3])
            roll = torch.tensor(last_wp_compact[3])
            pitch = torch.tensor(last_wp_compact[4])
            yaw = torch.tensor(last_wp_compact[5])
            q = quat_from_euler_xyz(roll, pitch, yaw)
            mat = quaternion_to_matrix(q.roll(1))
            mat_x = mat[:, 0]
            new_pos = pos + mat_x * append_dist

            waypoint_list.append(
                Waypoint(
                    index=last_wp.index + 1,
                    xyz=new_pos.tolist(),
                    rpy=last_wp.rpy,
                    length_y=last_wp.length_y,
                    length_z=last_wp.length_z,
                    gate=False,
                )
            )
        num_waypoints = len(waypoint_list)

        position = torch.zeros(num_envs, num_waypoints, 3)
        quaternion = torch.zeros(num_envs, num_waypoints, 4)
        width = torch.zeros(num_envs, num_waypoints)
        height = torch.zeros(num_envs, num_waypoints)
        gate_flag = torch.zeros(num_envs, num_waypoints, dtype=torch.int)
        gate_x_len_choice = torch.zeros(num_envs, num_waypoints, dtype=torch.int)
        gate_weight_choice = torch.zeros(num_envs, num_waypoints, dtype=torch.int)
        psi = torch.zeros(num_envs, num_waypoints - 1)
        theta = torch.zeros(num_envs, num_waypoints - 1)
        gamma = torch.zeros(num_envs, num_waypoints - 1)
        r = torch.zeros(num_envs, num_waypoints - 1)

        for i in range(num_waypoints):
            # [x, y, z, r, p, y, w, h, gate flag]
            # [0, 1, 2, 3, 4, 5, 6, 7, 8]
            data_src = waypoint_list[i].compact_data()
            position[:, i] = torch.tensor(data_src[:3])
            roll = torch.tensor([data_src[3]])
            pitch = torch.tensor([data_src[4]])
            yaw = torch.tensor([data_src[5]])
            quaternion[:, i] = quat_from_euler_xyz(roll, pitch, yaw)
            width[:, i] = data_src[6]
            height[:, i] = data_src[7]
            gate_flag[:, i] = int(data_src[8])

            # get psi, theta, gamma, and r
            if i > 0:
                # r
                start_pos = torch.tensor(waypoint_list[i - 1].xyz)
                end_pos = torch.tensor(waypoint_list[i].xyz)
                vec_r = end_pos - start_pos
                dist_r = torch.linalg.norm(vec_r)
                r[:, i - 1] = dist_r

                # psi, theta
                # dist_r == 0: psi = theta = 0
                if dist_r > 0:
                    start_q = quaternion[0, i - 1]
                    vec_r_b = quat_rotate_inverse(
                        start_q.unsqueeze(0), vec_r.unsqueeze(0)
                    ).squeeze()
                    if vec_r_b[0] == 0 and vec_r_b[1] == 0:
                        psi[:, i - 1] = 0
                        theta[:, i - 1] = torch.pi / 2 * torch.sign(vec_r_b[2])
                    else:
                        psi[:, i - 1] = torch.atan2(vec_r_b[1], vec_r_b[0])
                        theta[:, i - 1] = torch.atan2(
                            vec_r_b[2], torch.linalg.norm(vec_r_b[:2])
                        )

                # gamma
                end_q = quaternion[0, i]
                end_mat = quaternion_to_matrix(end_q.roll(1))
                end_x_axis = end_mat[:, 0]
                gamma[:, i - 1] = torch.acos(torch.sum(end_x_axis * vec_r) / dist_r)

        return cls(
            position,
            quaternion,
            width,
            height,
            gate_flag,
            gate_x_len_choice,
            gate_weight_choice,
            psi,
            theta,
            gamma,
            r,
        )

    @property
    def num_envs(self):
        return self.position.shape[0]

    @property
    def num_waypoints(self):
        return self.position.shape[1]

    @property
    def device(self):
        return self.position.device

    def to(self, device: str):
        self.position = self.position.to(device=device)
        self.quaternion = self.quaternion.to(device=device)
        self.width = self.width.to(device=device)
        self.height = self.height.to(device=device)
        self.gate_flag = self.gate_flag.to(device=device)
        self.gate_x_len_choice = self.gate_x_len_choice.to(device=device)
        self.gate_weight_choice = self.gate_weight_choice.to(device=device)
        self.psi = self.psi.to(device=device)
        self.theta = self.theta.to(device=device)
        self.gamma = self.gamma.to(device=device)
        self.r = self.r.to(device=device)

    def visualize(
        self,
        gym: Gym,
        envs: List[Env],
        viewer: Viewer,
        axes_len: float,
    ):
        axes = gymutil.AxesGeometry(axes_len)
        num_envs = len(envs)
        assert num_envs == self.num_envs

        for i in range(num_envs):
            for j in range(self.num_waypoints):
                x, y, z = self.position[i, j].tolist()
                qx, qy, qz, qw = self.quaternion[i, j].tolist()
                if j == 0:
                    box_color = (1.0, 0.0, 0.0)
                else:
                    box_color = (0.0, 1.0, 0.0)
                box = gymutil.WireframeBoxGeometry(
                    0.1,
                    float(self.width[i, j]),
                    float(self.height[i, j]),
                    color=box_color,
                )

                tf = Transform()
                tf.p = Vec3(x, y, z)
                tf.r = Quat(qx, qy, qz, qw)
                gymutil.draw_lines(axes, gym, viewer, envs[i], tf)
                gymutil.draw_lines(box, gym, viewer, envs[i], tf)

                if j < self.num_waypoints - 1:
                    x_next, y_next, z_next = self.position[i, j + 1].tolist()
                    p_next = Vec3(x_next, y_next, z_next)
                    line_color = Vec3(0.0, 1.0, 0.0)
                    gymutil.draw_line(tf.p, p_next, line_color, gym, viewer, envs[i])
