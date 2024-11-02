import isaacgym  # noqa
from isaacgymenvs.tasks.drone_racing.encoders.dce import (
    VAEImageEncoder,
    VAEImageEncoderConfig,
)

torch = None
import torch  # noqa


enc = VAEImageEncoder(VAEImageEncoderConfig())
inp = torch.rand(2, 270, 480, device="cuda")
out = enc.encode(inp)
recon = enc.decode(out)

print(enc.vae_model)
print(out.shape)
print(recon.shape)
print(enc.vae_model.inference_mode, enc.config.return_sampled_latent)
