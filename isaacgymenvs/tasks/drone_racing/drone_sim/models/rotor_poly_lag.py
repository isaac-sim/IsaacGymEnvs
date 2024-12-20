import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from isaacgymenvs.utils.torch_jit_utils import quaternion_to_matrix


@dataclass
class RotorPolyLagParams:
    """
    Default data source: https://store.tmotor.com/goods.php?id=1106
    and https://agilicious.readthedocs.io/en/latest/hardware/overview.html
    """

    # number of envs in parallel
    num_envs: int = 64

    # tensor device
    device: str = "cuda"

    # update period (seconds)
    dt: float = 1 / 500

    # number of rotors per env
    num_rotors: int = 4

    # rotor directions
    rotors_dir: List[float] = field(default_factory=lambda: [1, -1, -1, 1])

    # Time constant for rotor acceleration for first-order lag
    spinup_time_constant: float = 0.033

    # Time constant for rotor deceleration for first-order lag
    slowdown_time_constant: float = 0.033

    # Quadratic term of the polynomial model
    k_rpm_quadratic: float = -13421.95

    # Linear term of the polynomial model
    k_rpm_linear: float = 37877.42

    # The diagonal elements of the 3x3 diagonal inertia matrix
    rotor_diagonal_inertia: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 9.3575e-6]
    )

    # The quaternion (w, x, y, z) representing the same rotation as the principal axes matrix
    rotor_principle_axes_q: List[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0]
    )


class RotorPolyLag:

    def __init__(self, params: RotorPolyLagParams):
        self.params = params
        self.all_env_id = torch.arange(params.num_envs, device=params.device)

        # first-order lag param
        self.alpha_spinup = math.exp(-params.dt / params.spinup_time_constant)
        self.alpha_slowdown = math.exp(-params.dt / params.slowdown_time_constant)

        # rotor direction tensor
        self.rotor_dir = torch.tensor(params.rotors_dir, device=params.device)

        # init inertia matrices
        principle_axes_q = torch.tensor(
            params.rotor_principle_axes_q, device=params.device
        )
        principle_axes = quaternion_to_matrix(principle_axes_q)
        diagonal_inertia_mat = torch.diag(
            torch.tensor(params.rotor_diagonal_inertia, device=params.device)
        )
        rotated_inertia_mat = principle_axes @ diagonal_inertia_mat @ principle_axes.T
        self.rotor_inertia = torch.zeros(
            params.num_envs, params.num_rotors, 3, 3, device=params.device
        )
        self.rotor_inertia[:] = rotated_inertia_mat

        # init zero tensors
        self.rpm = torch.zeros(params.num_envs, params.num_rotors, device=params.device)
        self.rpm_ss = torch.zeros(
            params.num_envs, params.num_rotors, device=params.device
        )
        self.omega_dot = torch.zeros(
            params.num_envs, params.num_rotors, 3, device=params.device
        )
        self.force = torch.zeros(
            params.num_envs, params.num_rotors, 3, device=params.device
        )
        self.torque = torch.zeros(
            params.num_envs, params.num_rotors, 3, device=params.device
        )

    def reset(self, env_id: torch.Tensor = None):
        if env_id is None:
            env_id = self.all_env_id

        self.rpm[env_id, ...] = 0
        self.rpm_ss[env_id, ...] = 0
        self.omega_dot[env_id, ...] = 0
        self.force[env_id, ...] = 0
        self.torque[env_id, ...] = 0

    def compute(
        self, command: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main processing logic.

        Args:
            command: normalized rotor command in (num_envs, num_rotors).

        Returns:
            - Rotor RPM in (num_envs, num_rotors).
            - Force caused by motor dynamics in (num_envs, num_rotors, 3).
            - Torque caused by motor dynamics in (num_envs, num_rotors, 3).
        """

        # update rpm from first order lag
        rpm_to_ss = self.rpm_ss - self.rpm
        d_rpm = torch.where(
            rpm_to_ss >= 0,
            (1 - self.alpha_spinup) * rpm_to_ss,
            (1 - self.alpha_slowdown) * rpm_to_ss,
        )
        self.rpm += d_rpm

        # torque due to rotor acceleration
        self.omega_dot[:, :, -1] = (  # (num_envs, num_rotors)
            -d_rpm * 2 * torch.pi / 60 / self.params.dt * self.rotor_dir
        )
        self.torque[:, :] = torch.matmul(
            self.rotor_inertia,  # (num_envs, num_rotors, 3, 3)
            self.omega_dot.unsqueeze(-1),  # (num_envs, num_rotors, 3, 1)
        ).squeeze(3)

        # target RPM, as input to the first order lag system
        self.rpm_ss[:] = (
            self.params.k_rpm_quadratic * torch.pow(command, 2)
            + self.params.k_rpm_linear * command
        )

        return self.rpm, self.force, self.torque
