# Copyright (c) 2023, NVIDIA Corporation
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

"""IndustReal: class for peg insertion task.

Inherits IndustReal pegs environment class and Factory abstract task class (not enforced).

Trains a peg insertion policy with Simulation-Aware Policy Update (SAPU), SDF-Based Reward, and Sampling-Based Curriculum (SBC).

Can be executed with python train.py task=IndustRealTaskPegsInsert.
"""


import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils import torch_jit_utils


class IndustRealTaskPegsInsert(IndustRealEnvPegs, FactoryABCTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self._acquire_task_tensors()
        self.parse_controller_spec()

        # Get Warp mesh objects for SAPU and SDF-based reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        (
            self.wp_plug_meshes,
            self.wp_plug_meshes_sampled_points,
            self.wp_socket_meshes,
        ) = algo_utils.load_asset_meshes_in_warp(
            plug_files=self.plug_files,
            socket_files=self.socket_files,
            num_samples=self.cfg_task.rl.sdf_reward_num_samples,
            device=self.wp_device,
        )

        if self.viewer != None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        ppo_path = os.path.join(
            "train/IndustRealTaskPegsInsertPPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_pos_local = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    (self.cfg_task.env.socket_base_height + self.plug_grasp_offsets[i]),
                ]
                for i in range(self.num_envs)
            ],
            device=self.device,
        )
        self.gripper_goal_quat_local = self.identity_quat.clone()

        self.socket_top_pos_local = torch.tensor(
            [[0.0, 0.0, self.socket_heights[i]] for i in range(self.num_envs)],
            device=self.device,
        )
        self.socket_quat_local = self.identity_quat.clone()

        # Define keypoint tensors
        self.keypoint_offsets = (
            algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device)
            * self.cfg_task.rl.keypoint_scale
        )
        self.keypoints_plug = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_socket = torch.zeros_like(
            self.keypoints_plug, device=self.device
        )

        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )

        self.curr_max_disp = self.cfg_task.rl.initial_max_disp

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of gripper goal and top of socket in global frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_jit_utils.tf_combine(
            self.socket_quat,
            self.socket_pos,
            self.gripper_goal_quat_local,
            self.gripper_goal_pos_local,
        )

        self.socket_top_quat, self.socket_top_pos = torch_jit_utils.tf_combine(
            self.socket_quat,
            self.socket_pos,
            self.socket_quat_local,
            self.socket_top_pos_local,
        )

        # Add observation noise to socket pos
        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )
        socket_obs_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        socket_obs_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_gripper_goal_quat,
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine(
            self.noisy_socket_quat,
            self.noisy_socket_pos,
            self.gripper_goal_quat_local,
            self.gripper_goal_pos_local,
        )

        # Compute pos of keypoints on plug and socket in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(
                self.plug_quat,
                self.plug_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(
                self.socket_quat,
                self.socket_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        # Define observations (for actor)
        obs_tensors = [
            self.arm_dof_pos,  # 7
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                1
            ],  # 4
            self.pose_world_to_robot_base(
                self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat
            )[
                1
            ],  # 4
            noisy_delta_pos,
        ]  # 3

        # Define state (for critic)
        state_tensors = [
            self.arm_dof_pos,  # 7
            self.arm_dof_vel,  # 7
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                1
            ],  # 4
            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3
            self.pose_world_to_robot_base(
                self.gripper_goal_pos, self.gripper_goal_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.gripper_goal_pos, self.gripper_goal_quat
            )[
                1
            ],  # 4
            delta_pos,  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
            noisy_delta_pos - delta_pos,
        ]  # 3

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)
        self.states_buf = torch.cat(state_tensors, dim=-1)

        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        self.prev_rew_buf = self.rew_buf.clone()

        # SDF-Based Reward: Compute reward based on SDF distance
        sdf_reward = algo_utils.get_sdf_reward(
            wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
            asset_indices=self.asset_indices,
            plug_pos=self.plug_pos,
            plug_quat=self.plug_quat,
            plug_goal_sdfs=self.plug_goal_sdfs,
            wp_device=self.wp_device,
            device=self.device,
        )

        # SDF-Based Reward: Apply reward
        self.rew_buf[:] = self.cfg_task.rl.sdf_reward_scale * sdf_reward

        # SDF-Based Reward: Log reward
        self.extras["sdf_reward"] = torch.mean(self.rew_buf)

        # SAPU: Compute reward scale based on interpenetration distance
        low_interpen_envs, high_interpen_envs = [], []
        (
            low_interpen_envs,
            high_interpen_envs,
            sapu_reward_scale,
        ) = algo_utils.get_sapu_reward_scale(
            asset_indices=self.asset_indices,
            plug_pos=self.plug_pos,
            plug_quat=self.plug_quat,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
            wp_socket_meshes=self.wp_socket_meshes,
            interpen_thresh=self.cfg_task.rl.interpen_thresh,
            wp_device=self.wp_device,
            device=self.device,
        )

        # SAPU: For envs with low interpenetration, apply reward scale ("weight" step)
        self.rew_buf[low_interpen_envs] *= sapu_reward_scale

        # SAPU: For envs with high interpenetration, do not update reward ("filter" step)
        if len(high_interpen_envs) > 0:
            self.rew_buf[high_interpen_envs] = self.prev_rew_buf[high_interpen_envs]

        # SAPU: Log reward after scaling and adjustment from SAPU
        self.extras["sapu_adjusted_reward"] = torch.mean(self.rew_buf)

        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
            # Success bonus: Check which envs have plug engaged (partially inserted) or fully inserted
            is_plug_engaged_w_socket = algo_utils.check_plug_engaged_w_socket(
                plug_pos=self.plug_pos,
                socket_top_pos=self.socket_top_pos,
                keypoints_plug=self.keypoints_plug,
                keypoints_socket=self.keypoints_socket,
                cfg_task=self.cfg_task,
                progress_buf=self.progress_buf,
            )
            is_plug_inserted_in_socket = algo_utils.check_plug_inserted_in_socket(
                plug_pos=self.plug_pos,
                socket_pos=self.socket_pos,
                keypoints_plug=self.keypoints_plug,
                keypoints_socket=self.keypoints_socket,
                cfg_task=self.cfg_task,
                progress_buf=self.progress_buf,
            )

            # Success bonus: Compute reward scale based on whether plug is engaged with socket, as well as closeness to full insertion
            engagement_reward_scale = algo_utils.get_engagement_reward_scale(
                plug_pos=self.plug_pos,
                socket_pos=self.socket_pos,
                is_plug_engaged_w_socket=is_plug_engaged_w_socket,
                success_height_thresh=self.cfg_task.rl.success_height_thresh,
                device=self.device,
            )

            # Success bonus: Apply reward with reward scale
            self.rew_buf[:] += (
                engagement_reward_scale * self.cfg_task.rl.engagement_bonus
            )

            # Success bonus: Log success rate, ignoring environments with large interpenetration
            if len(high_interpen_envs) > 0:
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[
                    low_interpen_envs
                ]
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket_low_interpen.float()
                )
            else:
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket.float()
                )

            # SBC: Compute reward scale based on curriculum difficulty
            sbc_rew_scale = algo_utils.get_curriculum_reward_scale(
                cfg_task=self.cfg_task, curr_max_disp=self.curr_max_disp
            )

            # SBC: Apply reward scale (shrink negative rewards, grow positive rewards)
            self.rew_buf[:] = torch.where(
                self.rew_buf[:] < 0.0,
                self.rew_buf[:] / sbc_rew_scale,
                self.rew_buf[:] * sbc_rew_scale,
            )

            # SBC: Log current max downward displacement of plug at beginning of episode
            self.extras["curr_max_disp"] = self.curr_max_disp

            # SBC: Update curriculum difficulty based on success rate
            self.curr_max_disp = algo_utils.get_new_max_disp(
                curr_success=self.extras["insertion_successes"],
                cfg_task=self.cfg_task,
                curr_max_disp=self.curr_max_disp,
            )

    def _update_reset_buf(self):
        """Assign environments for reset if maximum episode length has been reached."""

        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka()

        # Close gripper onto plug
        self.disable_gravity()  # to prevent plug from falling
        self._reset_object()
        self._move_gripper_to_grasp_pose(
            sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        )
        self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        self.enable_gravity()

        # Get plug SDF in goal pose for SDF-based reward
        self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
            wp_plug_meshes=self.wp_plug_meshes,
            asset_indices=self.asset_indices,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_device=self.wp_device,
        )

        self._reset_buffers()

    def _reset_franka(self):
        """Reset DOF states, DOF torques, and DOF targets of Franka."""

        # Randomize DOF pos
        self.dof_pos[:] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos,
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
            ),
            dim=-1,
        ).unsqueeze(
            0
        )  # shape = (num_envs, num_dofs)

        # Stabilize Franka
        self.dof_vel[:, :] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[:, :] = 0.0
        self.ctrl_target_dof_pos = self.dof_pos.clone()
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

        # Set DOF state
        franka_actor_ids_sim = self.franka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_object(self):
        """Reset root state of plug and socket."""

        self._reset_socket()
        self._reset_plug(before_move_to_grasp=True)

    def _reset_socket(self):
        """Reset root state of socket."""

        # Randomize socket pos
        socket_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_noise_xy = socket_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.socket_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        socket_noise_z = torch.zeros(
            (self.num_envs), dtype=torch.float32, device=self.device
        )
        socket_noise_z_mag = (
            self.cfg_task.randomize.socket_pos_z_noise_bounds[1]
            - self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
        )
        socket_noise_z = (
            socket_noise_z_mag
            * torch.rand((self.num_envs), dtype=torch.float32, device=self.device)
            + self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
        )

        self.socket_pos[:, 0] = (
            self.robot_base_pos[:, 0]
            + self.cfg_task.randomize.socket_pos_xy_initial[0]
            + socket_noise_xy[:, 0]
        )
        self.socket_pos[:, 1] = (
            self.robot_base_pos[:, 1]
            + self.cfg_task.randomize.socket_pos_xy_initial[1]
            + socket_noise_xy[:, 1]
        )
        self.socket_pos[:, 2] = self.cfg_base.env.table_height + socket_noise_z

        # Randomize socket rot
        socket_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_rot_noise = socket_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.socket_rot_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        socket_rot_euler = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            + socket_rot_noise
        )
        socket_rot_quat = torch_utils.quat_from_euler_xyz(
            socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2]
        )
        self.socket_quat[:, :] = socket_rot_quat.clone()

        # Stabilize socket
        self.socket_linvel[:, :] = 0.0
        self.socket_angvel[:, :] = 0.0

        # Set socket root state
        socket_actor_ids_sim = self.socket_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(socket_actor_ids_sim),
            len(socket_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_plug(self, before_move_to_grasp):
        """Reset root state of plug."""

        if before_move_to_grasp:
            # Generate randomized downward displacement based on curriculum
            curr_curriculum_disp_range = (
                self.curr_max_disp - self.cfg_task.rl.curriculum_height_bound[0]
            )
            self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[
                0
            ] + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            # Generate plug pos noise
            self.plug_pos_xy_noise = 2 * (
                torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
                - 0.5
            )
            self.plug_pos_xy_noise = self.plug_pos_xy_noise @ torch.diag(
                torch.tensor(
                    self.cfg_task.randomize.plug_pos_xy_noise,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        self.plug_pos[:, :] = self.socket_pos.clone()
        self.plug_pos[:, 2] += self.socket_heights
        self.plug_pos[:, 2] -= self.curriculum_disp

        # Apply XY noise to plugs not partially inserted into sockets
        socket_top_height = self.socket_pos[:, 2] + self.socket_heights
        plug_partial_insert_idx = np.argwhere(
            self.plug_pos[:, 2].cpu().numpy() > socket_top_height.cpu().numpy()
        ).squeeze()
        self.plug_pos[plug_partial_insert_idx, :2] += self.plug_pos_xy_noise[
            plug_partial_insert_idx
        ]

        self.plug_quat[:, :] = self.identity_quat.clone()

        # Stabilize plug
        self.plug_linvel[:, :] = 0.0
        self.plug_angvel[:, :] = 0.0

        # Set plug root state
        plug_actor_ids_sim = self.plug_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(plug_actor_ids_sim),
            len(plug_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_buffers(self):
        """Reset buffers."""

        self.reset_buf[:] = 0
        self.progress_buf[:] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_centered_pos = (
            self.fingertip_centered_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_centered_quat
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _move_gripper_to_grasp_pose(self, sim_steps):
        """Define grasp pose for plug and move gripper to pose."""

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = self.plug_pos.clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += self.plug_grasp_offsets

        # Set target rot
        ctrl_target_fingertip_centered_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_centered_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2],
        )

        self.move_gripper_to_target_pose(
            gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            sim_steps=sim_steps,
        )

        # Reset plug in case it is knocked away by gripper movement
        self._reset_plug(before_move_to_grasp=False)
