from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class BodyDragPolyParams:
    """
    Default data source: https://agilicious.readthedocs.io/en/latest/hardware/overview.html
    """

    # number of parallel envs
    num_envs: int = 64

    # device to host tensor
    device: str = "cuda"

    # ISA air density [kg / m^3]
    air_density: float = 1.204

    # area pushing against air in body xyz direction (translational)
    a_trans: List[float] = field(default_factory=lambda: [1.5e-2, 1.5e-2, 3.0e-2])

    # quadratic translational drag coefficient in body xyz plane
    k_trans_quadratic: List[float] = field(default_factory=lambda: [1.04, 1.04, 1.04])

    # linear translational drag coefficient in body xyz plane
    k_trans_linear: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # equivalent area for calculating rotational drag
    a_rot: List[float] = field(default_factory=lambda: [1e-2, 1e-2, 1e-2])

    # quadratic rotational drag coefficient on body xyz axis
    k_rot_quadratic: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # linear rotational drag coefficient on body xyz axis
    k_rot_linear: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class BodyDragPoly:

    def __init__(self, params: BodyDragPolyParams):
        self.params = params

        self.a_trans = torch.tensor(params.a_trans, device=params.device)
        self.k_trans_quadratic = torch.tensor(
            params.k_trans_quadratic, device=params.device
        )
        self.k_trans_linear = torch.tensor(params.k_trans_linear, device=params.device)

        self.a_rot = torch.tensor(params.a_rot, device=params.device)
        self.k_rot_quadratic = torch.tensor(
            params.k_rot_quadratic, device=params.device
        )
        self.k_rot_linear = torch.tensor(params.k_rot_linear, device=params.device)

    def compute(
        self, lin_vel: torch.Tensor, ang_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main processing logic.

        Args:
            lin_vel: Linear velocity in body frame (num_envs, 3).
            ang_vel: Angular velocity in body frame (num_envs, 3).

        Returns:
            - Drag force (num_envs, 3).
            - Drag torque (num_envs, 3).
        """

        return _compute_script(
            self.params.air_density,
            self.a_trans,
            self.k_trans_quadratic,
            self.k_trans_linear,
            self.a_rot,
            self.k_rot_quadratic,
            self.k_rot_linear,
            lin_vel,
            ang_vel,
        )


@torch.jit.script
def _compute_script(
    air_density: float,
    a_trans: torch.Tensor,
    k_trans_quadratic: torch.Tensor,
    k_trans_linear: torch.Tensor,
    a_rot: torch.Tensor,
    k_rot_quadratic: torch.Tensor,
    k_rot_linear: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    force = (
        -0.5
        * air_density
        * a_trans
        * (k_trans_quadratic * lin_vel * torch.abs(lin_vel) + k_trans_linear * lin_vel)
    )
    torque = (
        -0.5
        * air_density
        * a_rot
        * (k_rot_quadratic * ang_vel * torch.abs(ang_vel) + k_rot_linear * ang_vel)
    )
    return force, torque
