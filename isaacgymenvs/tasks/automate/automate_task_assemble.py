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

"""AutoMate: class for specialist (part-specific) assembly policy training.

Inherits AutoMate environment class and Factory abstract task class (not enforced).

Can be executed with python train.py task=AutoMateTaskAssemble.

NOTE: to train a policy for a certain assembly, must collect disassembly paths 
for this assembly before training since the disassembly paths will be used to 
calculate reward during RL training.

"""


import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp
import json
import h5py

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
from isaacgymenvs.tasks.automate.automate_env import AutoMateEnv
from isaacgymenvs.utils import torch_jit_utils

import isaacgymenvs.tasks.factory.factory_control as fc

import isaacgymenvs.tasks.industreal.industreal_algo_utils as industreal_algo
import isaacgymenvs.tasks.automate.automate_algo_utils as automate_algo

from soft_dtw_cuda import SoftDTW

class AutoMateTaskAssemble(AutoMateEnv, FactoryABCTask):
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

        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()

        self.curriculum_height_bound, self.curriculum_height_step = self._get_curriculum_info(self.disassembly_dists)

        self._acquire_task_tensors()
        self.parse_controller_spec()

        # Load plug and socket meshes in warp (later used for SDF-based reward, Simulation-Aware Policy Update)
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_meshes_in_warp(self.plug_files, 
                                                                                                              self.socket_files, 
                                                                                                              self.cfg_task.rl.num_mesh_sample_points, 
                                                                                                              self.wp_device)

        # Create criterion for dynamic time warping (later used for imitation reward)
        self.soft_dtw_criterion = SoftDTW(use_cuda=True, gamma=self.cfg_task.rl.soft_dtw_gamma)

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
            "train/AutoMateTaskAssemblePPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _load_assembly_info(self):
        """Load grasp pose and disassembly distance for plugs in each environment."""

        plug_grasp_path = os.path.join(os.getcwd(), self.cfg_task.env.data_dir, self.cfg_task.env.plug_grasp_file)
        disassembly_dist_path = os.path.join(os.getcwd(), self.cfg_task.env.data_dir, self.cfg_task.env.disassembly_dist_file)
        
        in_file = open(plug_grasp_path, "r")
        plug_grasp_dict = json.load(in_file)
        plug_grasps = [ plug_grasp_dict[self.cfg_env.env.desired_subassemblies[self.asset_indices[i]]] for i in range(self.num_envs)]

        in_file = open(disassembly_dist_path, "r")
        disassembly_dist_dict = json.load(in_file)
        disassembly_dists = [ disassembly_dist_dict[self.cfg_env.env.desired_subassemblies[self.asset_indices[i]]] for i in range(self.num_envs)]

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _get_curriculum_info(self, disassembly_dists):
        """Calculate the ranges and step sizes for Sampling-based Curriculum (SBC) in each environment."""

        curriculum_height_bound = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        curriculum_height_step = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        curriculum_height_bound[:, 1] = disassembly_dists + self.cfg_task.rl.curriculum_freespace_range

        curriculum_height_step[:, 0] = curriculum_height_bound[:, 1] / self.cfg_task.rl.num_curriculum_step
        curriculum_height_step[:, 1] = - curriculum_height_step[:, 0] / 2.0

        return curriculum_height_bound, curriculum_height_step

    def _load_disassembly_data(self):
        """Load pre-collected disassembly trajectories (end-effector position only)."""

        # load disassembly data from json file
        log_filename = os.path.join(os.getcwd(), self.cfg_task.env.data_dir, self.cfg_task.env.desired_subassemblies[0]+'_disassembly_traj.json')
        
        disassembly_traj = json.load(open(log_filename))

        eef_pos_traj = []

        for i in range(len(disassembly_traj)):
            curr_ee_traj = np.asarray(disassembly_traj[i]['fingertip_centered_pos']).reshape((-1, 3))
            curr_ee_goal = np.asarray(disassembly_traj[i]['fingertip_centered_pos']).reshape((-1, 3))[0,:]
            
            # offset each trajectory to be relative to the goal
            eef_pos_traj.append(curr_ee_traj-curr_ee_goal)

        self.eef_pos_traj = torch.tensor(eef_pos_traj, dtype=torch.float32, device=self.device).squeeze()

    def apply_obs_noise(self):
        """Apply per-timestep observation noise on end-effector (gripper) goal pose."""
        
        # Add observation noise to socket pos
        noisy_socket_pos = torch.zeros_like(
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

        noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

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
        noisy_socket_quat = torch_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos = torch_jit_utils.tf_combine(noisy_socket_quat,
                                                                                               noisy_socket_pos,
                                                                                               self.plug_grasp_quat_local,
                                                                                               self.plug_grasp_pos_local)

        self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos = torch_jit_utils.tf_combine(self.noisy_gripper_goal_quat,
                                                                                               self.noisy_gripper_goal_pos,
                                                                                               self.robot_to_gripper_quat,
                                                                                               self.palm_to_finger_center)


    def _acquire_task_tensors(self):
        """Acquire tensors."""
        
        # Grasp pose tensors
        self.palm_to_finger_center = torch.tensor([0.0, 0.0, self.cfg_task.env.palm_to_finger_dist], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.robot_to_gripper_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.plug_grasp_pos_local = self.plug_grasps[:self.num_envs, :3]
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[:self.num_envs, 3:], -1, 1)

        self._load_disassembly_data()

        # Define keypoint tensors
        self.keypoint_offsets = (
            industreal_algo.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device)
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

        if self.cfg_task.env.if_eval:
            self.curr_max_disp = self.curriculum_height_bound[:, 1]
        else:
            self.curr_max_disp = torch.zeros((self.num_envs, ), dtype=torch.float32, device=self.device)

        self.plug_pos_noise = torch.zeros((self.num_envs, 3), device=self.device)
        self.plug_rot_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.insertion_successes = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        
        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_quat,
                                                                               self.plug_pos,
                                                                               self.plug_grasp_quat_local,
                                                                               self.plug_grasp_pos_local)

        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_grasp_quat,
                                                                                 self.plug_grasp_pos,
                                                                                 self.robot_to_gripper_quat,
                                                                                 self.palm_to_finger_center)

        self.gripper_goal_quat, self.gripper_goal_pos = torch_jit_utils.tf_combine(self.socket_quat,
                                                                                   self.socket_pos,
                                                                                   self.plug_grasp_quat_local,
                                                                                   self.plug_grasp_pos_local)

        self.gripper_goal_quat, self.gripper_goal_pos = torch_jit_utils.tf_combine(self.gripper_goal_quat,
                                                                                   self.gripper_goal_pos,
                                                                                   self.robot_to_gripper_quat,
                                                                                   self.palm_to_finger_center)

        # Compute pos of keypoints on gripper and plug in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_quat,
                                                                       self.socket_pos,
                                                                       self.identity_quat,
                                                                       keypoint_offset.repeat(self.num_envs, 1))[1]

            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_quat,
                                                                     self.plug_pos,
                                                                     self.identity_quat,
                                                                     keypoint_offset.repeat(self.num_envs, 1))[1]


    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        
        self._apply_actions_as_ctrl_targets(actions=self._actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True)


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

        self.apply_obs_noise()

        delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        obs_tensors = [
                       self.arm_dof_pos, # 7
                       self._pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0], # 3
                       self._pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1], # 4
                       self._pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0], # 3
                       self._pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1], # 4
                       noisy_delta_pos, 
                       ] 

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        state_tensors = [
            self.arm_dof_pos,  # 7
            self.arm_dof_vel,  # 7
            self._pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0], # 3
            self._pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1], # 4
            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3
            self._pose_world_to_robot_base(self.gripper_goal_pos, self.gripper_goal_quat)[0], # 3
            self._pose_world_to_robot_base(self.gripper_goal_pos, self.gripper_goal_quat)[1], # 4
            self._pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
            self._pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
            delta_pos,
        ]  # 3

        self.states_buf = torch.cat(state_tensors, dim=-1)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        
        self.prev_rew_buf = self.rew_buf.clone()

        # SDF-Based Reward: Calculate reward
        sdf_rwd = industreal_algo.get_sdf_reward(self.plug_sample_points,
                                                 self.asset_indices,
                                                 self.plug_pos,
                                                 self.plug_quat,
                                                 self.plug_goal_sdfs,
                                                 self.wp_device,
                                                 self.device)
        # SDF-Based Reward: Apply reward
        self.rew_buf[:] = self.cfg_task.rl.sdf_reward_scale * sdf_rwd
        
        # SDF-Based Reward: Log reward
        self.extras["sdf_reward"] = torch.mean(self.cfg_task.rl.sdf_reward_scale * sdf_rwd)

        # Imitation Reward: Calculate reward
        curr_eef_pos = self.fingertip_centered_pos - self.gripper_goal_pos # relative position instead of absolute position
        imitation_rwd = automate_algo.get_imitation_reward_from_dtw(self.eef_pos_traj, 
                                                                    curr_eef_pos, 
                                                                    self.prev_fingertip_centered_pos, 
                                                                    self.soft_dtw_criterion, 
                                                                    self.device)

        # Imitation Reward: Update end-effector trajectory window 
        self.prev_fingertip_centered_pos = torch.cat((self.prev_fingertip_centered_pos[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1)

        # Imitation Reward: Apply reward
        self.rew_buf[:] += self.cfg_task.rl.imitation_reward_scale * imitation_rwd

        # Imitation Reward: Log reward
        self.extras["imitation_reward"] = torch.mean(self.cfg_task.rl.imitation_reward_scale * imitation_rwd)

        # SAPU: Compute reward scale based on interpenetration distance
        low_interpen_envs, high_interpen_envs = [], []
        (
            low_interpen_envs,
            high_interpen_envs,
            sapu_reward_scale,
        ) = industreal_algo.get_sapu_reward_scale(
            asset_indices=self.asset_indices,
            plug_pos=self.plug_pos,
            plug_quat=self.plug_quat,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_plug_meshes_sampled_points=self.plug_sample_points,
            wp_socket_meshes=self.socket_mesh,
            interpen_thresh=self.cfg_task.rl.interpen_thresh,
            wp_device=self.wp_device,
            device=self.device
        )

        # SAPU: For envs with low interpenetration, apply reward scale ("weight" step)
        self.rew_buf[low_interpen_envs] *= sapu_reward_scale

        # SAPU: For envs with high interpenetration, do not update reward ("filter" step)
        if len(high_interpen_envs)>0:
            self.rew_buf[high_interpen_envs] = self.prev_rew_buf[high_interpen_envs]

        # SAPU: Log reward after scaling and adjustment from SAPU
        self.extras["sapu_adjusted_reward"] = torch.mean(self.rew_buf)

        # Only count reward that falls in the demonstration funnel
        reward_mask = automate_algo.get_reward_mask(self.eef_pos_traj, curr_eef_pos, self.cfg_task.rl.asset_tolerance)
        self.rew_buf[:] *= reward_mask

        is_plug_inserted_in_socket = automate_algo.check_plug_inserted_in_socket(
                                                                                plug_pos=self.plug_pos,
                                                                                socket_pos=self.socket_pos,
                                                                                curriculum_bound=self.curriculum_height_bound,
                                                                                keypoints_plug=self.keypoints_plug,
                                                                                keypoints_socket=self.keypoints_socket,
                                                                                cfg_task=self.cfg_task,
                                                                                progress_buf=self.progress_buf.clone().detach(),
                                                                            )

        self.insertion_successes = torch.logical_or(self.insertion_successes, is_plug_inserted_in_socket)

        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        if is_last_step:
            
            # Success bonus: Log success rate, ignoring environments with large interpenetration
            if len(high_interpen_envs) > 0:
                is_plug_inserted_in_socket_low_interpen = self.insertion_successes[
                    low_interpen_envs
                ]
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket_low_interpen.float()
                )
                self.insertion_successes[high_interpen_envs] = 0
            else:
                self.extras["insertion_successes"] = torch.mean(
                    self.insertion_successes.float()
                )

            self.rew_buf[:] += (self.insertion_successes * self.cfg_task.rl.engagement_bonus)

            # SBC: Compute reward scale based on curriculum difficulty
            sbc_rew_scale = automate_algo.get_curriculum_reward_scale(
                cfg_task=self.cfg_task, 
                curr_max_disp=self.curr_max_disp,
                curriculum_height_bound=self.curriculum_height_bound, 
                curriculum_height_step=self.curriculum_height_step,
            )
            self.extras["sbc_rew_scale"] = sbc_rew_scale

            # SBC: Apply reward scale (shrink negative rewards, grow positive rewards)
            self.rew_buf[:] = torch.where(
                self.rew_buf[:] < 0.0,
                self.rew_buf[:] / sbc_rew_scale,
                self.rew_buf[:] * sbc_rew_scale,
            )

            if not self.cfg_task.env.if_eval:
                # SBC: Update curriculum difficulty based on success rate
                self.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=self.extras["insertion_successes"],
                    cfg_task = self.cfg_task,
                    curriculum_height_bound=self.curriculum_height_bound, 
                    curriculum_height_step=self.curriculum_height_step, 
                    curr_max_disp=self.curr_max_disp,
                )
            
            # SBC: Log current max downward displacement of plug at beginning of episode
            self.extras["curr_max_disp"] = torch.mean(self.curr_max_disp)

            print("Insertion Success: ", self.extras["insertion_successes"].item())

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)

        # Temporarily disable gravity to prevent plugs from dropping before being grasped
        self.disable_gravity() 

        self._reset_object(env_ids)

        self._move_gripper_to_plug_grasp_pose(env_ids, mode='pre_grasp', sim_steps=self.cfg_task.env.move_gripper_sim_steps)

        self._move_gripper_to_plug_grasp_pose(env_ids, mode='grasp', sim_steps=self.cfg_task.env.move_gripper_sim_steps)

        # Reset plug again in case the plug collided with the end-effector during gripper movement
        self._reset_plug(env_ids)

        self.simulate_and_refresh()

        self.close_gripper(sim_steps=self.cfg_task.env.close_gripper_sim_steps)

        self.enable_gravity()

        # Generate goal SDFs using plug meshes at plug goal pose, later used for SDF-based reward
        self.plug_goal_sdfs = industreal_algo.get_plug_goal_sdfs(self.plug_mesh, self.asset_indices, self.socket_pos, self.socket_quat, self.wp_device)

        # Initialize end-effector trajectory with given window size, later used for imitation reward
        prev_fingertip_centered_pos = (self.fingertip_centered_pos-self.gripper_goal_pos).unsqueeze(1) # (num_envs, 1, 3)
        self.prev_fingertip_centered_pos = torch.repeat_interleave(prev_fingertip_centered_pos, 
                                                                    self.cfg_task.rl.num_point_robot_traj, 
                                                                    dim=1) # (num_envs, num_point_robot_traj, 3)


        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.insertion_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka 
        self.dof_pos[env_ids] = torch.cat(
                (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
                 torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
                 torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
                dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)

        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
                                                gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                len(self.franka_actor_ids_sim))

        self.simulate_and_refresh()

    def _reset_object(self, env_ids):
        """Reset plug and socket."""

        self._reset_socket(env_ids)
        self._generate_plug_disp_and_noise(env_ids)
        self._reset_plug(env_ids)

    def _reset_socket(self, env_ids):
        """Reset root state of socket."""

        # Randomize socket position 
        self.socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        self.socket_noise_xy = self.socket_noise_xy @ torch.diag(torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, dtype=torch.float32, device=self.device))
        self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.robot_base_pos[env_ids, 0] + self.cfg_task.randomize.socket_pos_xy_initial[0] + self.socket_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.robot_base_pos[env_ids, 1] + self.cfg_task.randomize.socket_pos_xy_initial[1] + self.socket_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_base.env.table_height + self.cfg_task.env.socket_z_offset

        # Set socket orientation to be upright
        self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        # Set socket velocities to be zero
        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        socket_actor_ids_sim = self.socket_actor_ids_sim[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(socket_actor_ids_sim[env_ids]),
                                                     len(socket_actor_ids_sim[env_ids]))
        self.simulate_and_refresh()


    def _reset_plug(self, env_ids):
        """Reset root state of plug."""

        # Randomize plug position relative to the socket
        self.root_pos[env_ids, self.plug_actor_id_env, :] = self.root_pos[env_ids, self.socket_actor_id_env, :]
        self.root_pos[env_ids, self.plug_actor_id_env, 2] += self.curriculum_disp[env_ids]

        # Find the plugs initialized outside of the socket
        plug_engage_dist = self.curriculum_height_bound[:,1]-self.cfg_task.rl.curriculum_freespace_range
        plug_not_engaged = torch.nonzero(self.curriculum_disp>plug_engage_dist)
        
        # Only randomize the plugs that are not partially engaged with the sockets 
        self.root_pos[plug_not_engaged, self.plug_actor_id_env, :2] += self.plug_pos_noise[env_ids][plug_not_engaged, :2]
        self.root_quat[plug_not_engaged, self.plug_actor_id_env] = self.plug_rot_quat[env_ids][plug_not_engaged, :]

        # Set plug orientation to be upright
        self.root_quat[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        # Set plug velocities to be zero
        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        multi_env_ids_int32 = self.plug_actor_ids_sim[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                     len(multi_env_ids_int32))

        self.simulate_and_refresh()

    def _generate_plug_disp_and_noise(self, env_ids):
        """Generate displacement for SBC and plug randomization."""
        
        # Generate randomized displacement along z-axis based on curriculum
        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp # shape: (num_envs, 1), different disp range for each env
        self.curriculum_disp = self.curriculum_height_bound[:, 1] - curr_curriculum_disp_range * (torch.rand((self.num_envs,), dtype=torch.float32, device=self.device))

        # Generate randomized plug position
        plug_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        self.plug_pos_noise[:] = plug_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.plug_pos_noise, device=self.device))

        # Generate randomized plug rotation
        plug_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        plug_rot_noise = plug_rot_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.plug_rot_noise, dtype=torch.float32, device=self.device))
        plug_rot_euler = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        plug_rot_euler += plug_rot_noise
        self.plug_rot_quat[:] = torch_utils.quat_from_euler_xyz(plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2])

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _move_gripper_to_eef_pose(self, env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log, close_gripper):
        """Move end-effector to a given pose specifed by (ctrl_tgt_pos, ctrl_tgt_quat)."""

        self.ctrl_target_fingertip_centered_pos[env_ids] = ctrl_tgt_pos[env_ids]
        self.ctrl_target_fingertip_centered_quat[env_ids] = ctrl_tgt_quat[env_ids]

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            if if_log:
                self._log_robot_state_per_timestep()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[env_ids, :6] = delta_hand_pose[env_ids]

            if close_gripper:
                self._apply_actions_as_ctrl_targets(actions=actions,
                                                    ctrl_target_gripper_dof_pos=0.0,
                                                    do_scale=False)
            else:
                self._apply_actions_as_ctrl_targets(actions=actions,
                                                    ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                                    do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
                                                gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                len(self.franka_actor_ids_sim))

    def _move_gripper_to_plug_grasp_pose(self, env_ids, mode, sim_steps):
        """Move end-effector to plug grasp pose."""

        ctrl_tgt_pos = torch.empty_like(self.plug_grasp_pos).copy_(self.plug_grasp_pos)

        if mode=='grasp':
            ctrl_tgt_quat = torch.empty_like(self.plug_grasp_quat).copy_(self.plug_grasp_quat)

        elif mode=='pre_grasp':
            ctrl_tgt_pos[:, 2] += self.cfg_task.env.plug_pregrasp_offset
            ctrl_tgt_quat = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log=False, close_gripper=False)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""
        
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _pose_world_to_robot_base(self, pos, quat):
        """Convert pose from world frame to robot base frame."""

        robot_base_transform_inv = torch_utils.tf_inverse(self.robot_base_quat, self.robot_base_pos)
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(robot_base_transform_inv[0],
                                                                       robot_base_transform_inv[1],
                                                                       quat,
                                                                       pos)
        return pos_in_robot_base, quat_in_robot_base
        