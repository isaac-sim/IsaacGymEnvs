from dataclasses import dataclass, field
from typing import List

import torch
from torch import pi

from isaacgym import torch_utils
from .waypoint_data import WaypointData


@dataclass
class WaypointGeneratorParams:
    num_envs: int = 64

    num_waypoints: int = 4

    num_gate_x_lens: int = 2

    num_gate_weights: int = 2

    gate_weight_max: float = 0.3

    fixed_waypoint_id: int = 1

    fixed_waypoint_position: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )


@dataclass
class RandWaypointOptions:

    wp_size_min: float = 1.0

    wp_size_max: float = 3.0

    init_roll_max: float = pi / 6

    init_pitch_max: float = pi / 6

    init_yaw_max: float = pi / 1

    psi_max: float = pi / 2

    theta_max: float = pi / 4

    alpha_max: float = pi / 1

    gamma_max: float = pi / 6

    r_min: float = 2.0

    r_max: float = 20.0

    # -1: random, 0: force 0, 1: force 1, other values raise error
    force_gate_flag: int = -1

    # if True, tracks for multiple envs are the same
    same_track: bool = False


class WaypointGenerator:

    def __init__(self, params: WaypointGeneratorParams):
        self.params = params
        self.anchor_pos = torch.tensor(params.fixed_waypoint_position)

    def compute(self, options: RandWaypointOptions) -> WaypointData:
        """
        Generates an instance of ``WaypointData`` randomly according to options.

        Args:
            options: An instance of ``RandomWaypointOptions`` containing min and max values for random sampling.

        Returns:
            - An instance of ``WaypointData``.
        """

        num_envs = self.params.num_envs
        if options.same_track:
            num_envs = 1

        # waypoint width and height
        wp_size_range = options.wp_size_max - options.wp_size_min
        width = (
            torch.rand(num_envs, self.params.num_waypoints) * wp_size_range
            + options.wp_size_min
        )
        height = (
            torch.rand(num_envs, self.params.num_waypoints) * wp_size_range
            + options.wp_size_min
        )

        # associated gate params
        assert -1 <= options.force_gate_flag <= 1
        gate_flag = torch.randint(0, 2, (num_envs, self.params.num_waypoints))
        if not options.force_gate_flag == -1:
            gate_flag[:] = options.force_gate_flag
        gate_x_len_id = torch.randint(
            0,
            self.params.num_gate_x_lens,
            (num_envs, self.params.num_waypoints),
        )
        gate_weight_id = torch.randint(
            0,
            self.params.num_gate_weights,
            (num_envs, self.params.num_waypoints),
        )

        # initial waypoint attitude
        init_roll = (
            torch.rand(num_envs) * 2 * options.init_roll_max - options.init_roll_max
        )
        init_pitch = (
            torch.rand(num_envs) * 2 * options.init_pitch_max - options.init_pitch_max
        )
        init_yaw = (
            torch.rand(num_envs) * 2 * options.init_yaw_max - options.init_yaw_max
        )

        # waypoint relative pose params
        psi = (
            torch.rand(num_envs, self.params.num_waypoints - 1) * 2 * options.psi_max
            - options.psi_max
        )
        theta = (
            torch.rand(num_envs, self.params.num_waypoints - 1) * 2 * options.theta_max
            - options.theta_max
        )
        alpha = (
            torch.rand(num_envs, self.params.num_waypoints - 1) * 2 * options.alpha_max
            - options.alpha_max
        )
        gamma = torch.rand(num_envs, self.params.num_waypoints - 1) * options.gamma_max

        r_wp = (
            width**2 + height**2
        ) ** 0.5 / 2 + gate_flag * 2**0.5 * self.params.gate_weight_max
        r_lb = r_wp[:, :-1] + r_wp[:, 1:]
        r_lb.clamp_(min=options.r_min)
        r_ub = options.r_max * torch.ones_like(r_lb)
        r_ub.clamp_(min=r_lb)
        r_range = r_ub - r_lb
        r = torch.rand(num_envs, self.params.num_waypoints - 1) * r_range + r_lb

        # calculate pose
        pos = torch.zeros(num_envs, self.params.num_waypoints, 3)
        quat = torch.zeros(num_envs, self.params.num_waypoints, 4)
        for i in range(self.params.num_waypoints):
            if i == 0:
                quat[:, i] = torch_utils.quat_from_euler_xyz(
                    init_roll, init_pitch, init_yaw
                )
            else:
                psi_f = psi[:, i - 1]
                theta_f = theta[:, i - 1]
                alpha_f = alpha[:, i - 1]
                gamma_f = gamma[:, i - 1]
                zeros_f = torch.zeros_like(gamma_f)

                q_psi_theta = torch_utils.quat_from_euler_xyz(zeros_f, -theta_f, psi_f)
                q_alpha = torch_utils.quat_from_euler_xyz(alpha_f, zeros_f, zeros_f)
                q_gamma = torch_utils.quat_from_euler_xyz(zeros_f, gamma_f, zeros_f)

                q0 = quat[:, i - 1]  # (num_envs, 4)
                q1 = torch_utils.quat_mul(q0, q_psi_theta)
                q2 = torch_utils.quat_mul(q1, q_alpha)
                q3 = torch_utils.quat_mul(q2, q_gamma)
                quat[:, i] = q3

                r_vec = torch.zeros(num_envs, 3)  # body frame
                r_vec[:, 0] = r[:, i - 1]
                r_vec_rotated = torch_utils.quat_rotate(q1, r_vec)  # world frame
                pos[:, i] = pos[:, i - 1] + r_vec_rotated

        # anchor waypoints
        offset = (
            self.anchor_pos.unsqueeze(0) - pos[:, self.params.fixed_waypoint_id]
        )  # (num_envs, 3)
        pos += offset.unsqueeze(1)

        if num_envs == 1:
            pos = pos.expand(self.params.num_envs, -1, -1)
            quat = quat.expand(self.params.num_envs, -1, -1)
            width = width.expand(self.params.num_envs, -1)
            height = height.expand(self.params.num_envs, -1)
            gate_flag = gate_flag.expand(self.params.num_envs, -1)
            gate_x_len_id = gate_x_len_id.expand(self.params.num_envs, -1)
            gate_weight_id = gate_weight_id.expand(self.params.num_envs, -1)
            psi = psi.expand(self.params.num_envs, -1)
            theta = theta.expand(self.params.num_envs, -1)
            gamma = gamma.expand(self.params.num_envs, -1)
            r = r.expand(self.params.num_envs, -1)

        return WaypointData(
            position=pos,
            quaternion=quat,
            width=width,
            height=height,
            gate_flag=gate_flag,
            gate_x_len_choice=gate_x_len_id,
            gate_weight_choice=gate_weight_id,
            psi=psi,
            theta=theta,
            gamma=gamma,
            r=r,
        )
