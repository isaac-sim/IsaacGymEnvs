import os
from dataclasses import dataclass
from typing import Tuple

from typing_extensions import LiteralString


@dataclass
class VAEImageEncoderConfig:
    use_vae: bool = True
    latent_dims: int = 64
    model_file: LiteralString = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pth",
        "ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth",
    )
    model_folder: LiteralString = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pth"
    )
    image_res: Tuple[float, float] = (270, 480)
    interpolation_mode: str = "nearest"
    return_sampled_latent: bool = True  # TODO: why True?
