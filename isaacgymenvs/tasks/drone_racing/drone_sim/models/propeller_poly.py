from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class PropellerPolyParams:
    """
    Default data source: https://store.tmotor.com/goods.php?id=1106
    and https://agilicious.readthedocs.io/en/latest/hardware/overview.html
    """

    # number of parallel envs
    num_envs: int = 64

    # tensor device
    device: str = "cuda"

    # number of propellers per environment
    num_props: int = 4

    # propeller directions relative to body z axis (FRD)
    prop_dir: List[int] = field(default_factory=lambda: [1, -1, -1, 1])

    # quadratic coefficient for force calculation
    k_force_quadratic: float = 2.1549e-08

    # linear coefficient for force calculation
    k_force_linear: float = -4.5101e-05

    # quadratic coefficient for torque calculation
    k_torque_quadratic: float = 2.1549e-08 * 0.022

    # linear coefficient for torque calculation
    k_torque_linear: float = -4.5101e-05 * 0.022


class PropellerPoly:

    def __init__(self, params: PropellerPolyParams):
        self.params = params

        self.torque_dir = -torch.tensor(params.prop_dir, device=params.device)

        self.force = torch.zeros(
            params.num_envs, params.num_props, 3, device=params.device
        )
        self.torque = torch.zeros(
            params.num_envs, params.num_props, 3, device=params.device
        )

    def compute(self, rpm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main processing logic.

        Args:
            rpm: propeller RPM tensor in (num_envs, num_propellers).

        Returns:
            - Force caused by propeller in (num_envs, num_propellers, 3).
            - Torque caused by propeller in (num_envs, num_propellers, 3).
        """

        self.force[:, :, 2], self.torque[:, :, 2] = _compute_script(
            self.params.k_force_quadratic,
            self.params.k_force_linear,
            self.params.k_torque_quadratic,
            self.params.k_torque_linear,
            self.torque_dir,
            rpm,
        )

        return self.force, self.torque


@torch.jit.script
def _compute_script(
    k_force_quadratic: float,
    k_force_linear: float,
    k_torque_quadratic: float,
    k_torque_linear: float,
    torque_dir: torch.Tensor,
    rpm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    f = (-1) * (k_force_quadratic * torch.pow(rpm, 2) + k_force_linear * rpm)
    t = (k_torque_quadratic * torch.pow(rpm, 2) + k_torque_linear * rpm) * torque_dir
    return f, t
