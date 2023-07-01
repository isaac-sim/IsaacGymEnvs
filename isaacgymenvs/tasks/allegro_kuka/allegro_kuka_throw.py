# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
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

from typing import List

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import tolerance_successes_objective


class AllegroKukaThrow(AllegroKukaBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.bucket_asset = self.bucket_pose = None
        self.bucket_object_indices = []

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _object_keypoint_offsets(self):
        """Throw task uses only a single object keypoint since we do not care about object orientation."""
        return [[0, 0, 0]]

    def _load_additional_assets(self, object_asset_root, arm_pose):
        """
        returns: tuple (num_rigid_bodies, num_shapes)
        """
        bucket_asset_options = gymapi.AssetOptions()
        bucket_asset_options.disable_gravity = False
        bucket_asset_options.fix_base_link = True
        bucket_asset_options.collapse_fixed_joints = True
        bucket_asset_options.vhacd_enabled = True
        bucket_asset_options.vhacd_params = gymapi.VhacdParams()
        bucket_asset_options.vhacd_params.resolution = 500000
        bucket_asset_options.vhacd_params.max_num_vertices_per_ch = 32
        bucket_asset_options.vhacd_params.min_volume_per_ch = 0.001
        self.bucket_asset = self.gym.load_asset(
            self.sim, object_asset_root, self.asset_files_dict["bucket"], bucket_asset_options
        )

        self.bucket_pose = gymapi.Transform()
        self.bucket_pose.p = gymapi.Vec3()
        self.bucket_pose.p.x = arm_pose.p.x - 0.6
        self.bucket_pose.p.y = arm_pose.p.y - 1
        self.bucket_pose.p.z = arm_pose.p.z + 0.45

        bucket_rb_count = self.gym.get_asset_rigid_body_count(self.bucket_asset)
        bucket_shapes_count = self.gym.get_asset_rigid_shape_count(self.bucket_asset)
        print(f"Bucket rb {bucket_rb_count}, shapes {bucket_shapes_count}")

        return bucket_rb_count, bucket_shapes_count

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        bucket_handle = self.gym.create_actor(
            env_ptr, self.bucket_asset, self.bucket_pose, "bucket_object", env_idx, 0, 0
        )
        bucket_object_idx = self.gym.get_actor_index(env_ptr, bucket_handle, gymapi.DOMAIN_SIM)
        self.bucket_object_indices.append(bucket_object_idx)

    def _after_envs_created(self):
        self.bucket_object_indices = to_torch(self.bucket_object_indices, dtype=torch.long, device=self.device)

    def _reset_target(self, env_ids: Tensor) -> None:
        # whether we place the bucket to the left or to the right of the table
        left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
        x_pos = torch.where(
            left_right_random > 0, 0.5 * torch.ones_like(left_right_random), -0.5 * torch.ones_like(left_right_random)
        )
        x_pos += torch.sign(left_right_random) * torch_rand_float(0, 0.4, (len(env_ids), 1), device=self.device)
        # y_pos = torch_rand_float(-0.6, 0.4, (len(env_ids), 1), device=self.device)
        y_pos = torch_rand_float(-1.0, 0.7, (len(env_ids), 1), device=self.device)
        z_pos = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device)
        self.root_state_tensor[self.bucket_object_indices[env_ids], 0:1] = x_pos
        self.root_state_tensor[self.bucket_object_indices[env_ids], 1:2] = y_pos
        self.root_state_tensor[self.bucket_object_indices[env_ids], 2:3] = z_pos

        self.goal_states[env_ids, 0:1] = x_pos
        self.goal_states[env_ids, 1:2] = y_pos
        self.goal_states[env_ids, 2:3] = z_pos + 0.05

        # we also reset the object to its initial position
        self.reset_object_pose(env_ids)

        # since we put the object back on the table, also reset the lifting reward
        self.lifted_object[env_ids] = False

        object_indices_to_reset = [self.bucket_object_indices[env_ids], self.object_indices[env_ids]]
        self.deferred_set_actor_root_state_tensor_indexed(object_indices_to_reset)

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return [self.bucket_object_indices[env_ids]]

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective
