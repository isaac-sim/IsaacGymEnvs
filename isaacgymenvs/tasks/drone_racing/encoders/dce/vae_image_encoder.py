# BSD 3-Clause License
#
# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# https://github.com/ntnu-arl/aerial_gym_simulator.git
# commit 070391cc30d92b76dcd3e4e41a49c8d1b60080ae (HEAD -> main, origin/main, origin/HEAD)
# Author: Mihir Kulkarni <mihirk284@gmail.com>
# Date:   Fri Jul 26 14:47:14 2024 +0200
#
#     Fix bug that misplaced function arguments
#
#     Signed-off-by: Mihir Kulkarni <mihirk284@gmail.com>

import os

import torch

from .VAE import VAE


def clean_state_dict(state_dict):
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        # combine module path with model file name
        weight_file_path = os.path.join(
            self.config.model_folder, self.config.model_file
        )
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))  # noqa
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():
            # need to squeeze 0th dimension and unsqueeze 1st dimension to make it work with the VAE
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} "
                    f"does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims
