from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class WrenchSumParams:
    num_envs: int = 64
    device: str = "cuda"
    # wrench application positions in body FRD frame
    # the order should match that of the wrench
    # default values mean there are 4 positions of application
    # the n-th position is (position_x[n], position_y[n], position_z[n]) w.r.t. body FRD frame
    num_positions: int = 4
    position_x: List[float] = field(
        default_factory=lambda: [-0.078665, 0.078665, -0.078665, 0.078665]
    )
    position_y: List[float] = field(
        default_factory=lambda: [0.097143, 0.097143, -0.097143, -0.097143]
    )
    position_z: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


class WrenchSum:

    def __init__(self, params: WrenchSumParams):
        self.params = params

        self.r = torch.zeros(
            params.num_envs, params.num_positions, 3, device=params.device
        )
        self.r[:] = torch.tensor(
            [params.position_x, params.position_y, params.position_z],
            device=params.device,
        ).T

    def compute(
        self, force: torch.Tensor, torque: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes total force and torque from scattered wrench.

        The total force is the sum of all scattered forces.
        The total torque is the sum of all scattered torques plus force-induced torques.

        Args:
            force: force applied to scattered positions, (num_envs, num_positions, 3).
            torque: torque applied to scattered positions, (num_envs, num_positions, 3).


        Returns:
            - Total force tensor (num_envs, 3) to be applied to the body frame origin.
            - Total torque tensor (num_envs, 3) to be applied to the body frame origin.
        """

        return _compute_script(self.r, force, torque)


@torch.jit.script
def _compute_script(
    r: torch.Tensor, force: torch.Tensor, torque: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_force = force.sum(dim=1)
    total_torque = torque.sum(dim=1) + torch.cross(r, force, 2).sum(dim=1)
    return total_force, total_torque
