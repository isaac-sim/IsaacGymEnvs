import os
from typing import List

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_juggle_base import AllegroKukaJuggleBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import tolerance_curriculum, tolerance_successes_objective

class AllegroKukaJuggle(AllegroKukaJuggleBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.goal_object_indices = []
        self.goal_assets = []

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _object_keypoint_offsets(self):
        return [
            [0, 0, 0],
        ]

    def _load_additional_assets(self, object_asset_root, arm_pose):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        self.goal_assets = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            goal_asset_ = self.gym.load_asset(self.sim, object_asset_dir, object_asset_fname, object_asset_options)
            self.goal_assets.append(goal_asset_)
        goal_rb_count = self.gym.get_asset_rigid_body_count(
            self.goal_assets[0]
        )  # assuming all of them have the same rb count
        goal_shapes_count = self.gym.get_asset_rigid_shape_count(
            self.goal_assets[0]
        )  # assuming all of them have the same rb count

        return goal_rb_count, goal_shapes_count

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        # self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
        # self.goal_displacement_tensor = to_torch(
        #     [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device
        # )
        # goal_start_pose = gymapi.Transform()
        # goal_start_pose.p = self.object_start_pose.p + self.goal_displacement
        # goal_start_pose.p.z -= 0.04
        #
        # goal_asset = self.goal_assets[object_asset_idx]
        # goal_handle = self.gym.create_actor(
        #     env_ptr, goal_asset, goal_start_pose, "goal_object", env_idx + self.num_envs, 0, 0
        # )
        # goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
        # self.goal_object_indices.append(goal_object_idx)
        #
        # if self.object_type != "block":
        #     self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
        pass

    def _after_envs_created(self):
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    # def _extra_reset_rules(self, resets):
    #     # hand far from the object
    #     resets = torch.where(
    #         self.curr_fingertip_distances.max(dim=-1).values > 1.5, torch.ones_like(self.reset_buf), resets
    #     )
    #     return resets

    def _reset_target(self, env_ids: Tensor) -> None:

        # target_volume_origin = self.target_volume_origin
        # target_volume_extent = self.target_volume_extent
        #
        # target_volume_min_coord = target_volume_origin + target_volume_extent[:, 0]
        # target_volume_max_coord = target_volume_origin + target_volume_extent[:, 1]
        # target_volume_size = target_volume_max_coord - target_volume_min_coord
        #
        # rand_pos_floats = torch_rand_float(0.0, 1.0, (len(env_ids), 3), device=self.device)
        # target_coords = target_volume_min_coord + rand_pos_floats * target_volume_size
        # target_coords[:, 0] = 0
        # target_coords[:, 1] = 0.05
        # target_coords[:, 2] = 1.4
        # self.goal_states[env_ids, 0:3] = target_coords
        #
        # self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]
        #
        # new_rot = self.get_random_quat(env_ids)
        # new_rot[:, 0:3] = 0.0
        # new_rot[:, 3] = 1.0
        # self.goal_states[env_ids, 3:7] = new_rot
        #
        # self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        # self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
        #     self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
        # )
        #
        # object_indices_to_reset = [self.goal_object_indices[env_ids]]
        # self.deferred_set_actor_root_state_tensor_indexed(object_indices_to_reset)
        pass

    # def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
    #     return [self.goal_object_indices[env_ids]]

    # def _extra_curriculum(self):
    #     self.success_tolerance, self.last_curriculum_update = tolerance_curriculum(
    #         self.last_curriculum_update,
    #         self.frame_since_restart,
    #         self.tolerance_curriculum_interval,
    #         self.prev_episode_successes,
    #         self.success_tolerance,
    #         self.initial_tolerance,
    #         self.target_tolerance,
    #         self.tolerance_curriculum_increment,
    #     )

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective
