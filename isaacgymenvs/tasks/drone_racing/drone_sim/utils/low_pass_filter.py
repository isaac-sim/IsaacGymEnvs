from dataclasses import dataclass

import torch


@dataclass
class FirstOrderLowPassFilterParams:
    device: str = "cuda"
    dim: torch.Size = torch.Size([64, 1])
    dt: float = 0.001
    cutoff_frequency: float = 100.0
    initial_value: float = 0.0


class FirstOrderLowPassFilter:

    def __init__(self, params: FirstOrderLowPassFilterParams):
        self.params = params
        self.all_env_id = torch.arange(params.dim[0], device=params.device)

        self.alpha = 1 / (
            1 + 1 / (2 * torch.pi * params.cutoff_frequency * self.params.dt)
        )
        self.output = params.initial_value * torch.ones(
            self.params.dim, device=self.params.device
        )

    def reset(self, env_id: torch.Tensor = None, initial_value: float = None):
        if env_id is None:
            env_id = self.all_env_id

        if initial_value is None:
            initial_value = self.params.initial_value

        self.output[env_id, ...] = initial_value

    def get_output(self) -> torch.Tensor:
        return self.output

    def update(self, data: torch.Tensor):
        self.output += self.alpha * (data - self.output)
