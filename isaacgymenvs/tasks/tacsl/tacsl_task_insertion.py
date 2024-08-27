# Copyright (c) 2024, NVIDIA Corporation
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

"""TacSL: Class for insertion task.

Inherits insertion environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskInsertion

Only the environment is provided; training a successful RL policy is an open research problem left to the user.
"""

import hydra
import numpy as np
import omegaconf
import torch

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.tasks.tacsl.tacsl_env_insertion import TacSLEnvInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_image_augmentation import TacSLTaskImageAugmentation
from isaacgymenvs.utils import torch_jit_utils


class TacSLTaskInsertion(TacSLTaskImageAugmentation, TacSLEnvInsertion, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self._acquire_task_tensors()

        if self.cfg_task.env.use_isaac_gym_tactile:
            assert self.cfg_task.env.use_gelsight, "shear force currently works only with gelsight fingers"
            # Open finger to render nominal tactile sensor
            self.initialize_franka_robot_open_hand()
            # Initialize tactile sensors
            self.initialize_tactile_rgb_camera()

        if self.cfg_task.env.use_shear_force:
            assert self.cfg_task.env.use_gelsight, "shear force currently works only with gelsight fingers"
            num_divs = [self.cfg_task.env.num_shear_rows, self.cfg_task.env.num_shear_cols]
            self.initialize_penalty_based_tactile(num_divs=num_divs)

        if self.cfg_task.env.task_type == 'placement':
            # The placement task moves the peg to the tip of the placement pad, no insertion
            self.cfg_task.rl.insertion_frac = 0.0

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='tacsl_task_insertion')

        self.set_friction_damping_params(joint_friction=self.cfg_task.env.joint_friction,
                                         joint_damping=self.cfg_task.env.joint_damping)

        self.image_obs_keys = [k for k, v in self.obs_dims.items() if len(v) > 2 and 'force_field' not in k]
        self.init_image_augmentation()

        self.reset_idx(torch.arange(self.num_envs))

    def initialize_franka_robot_open_hand(self):
        """
        Initialize Franka robot to default dof position.
        """
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)

        self.ctrl_target_dof_pos[:] = self.dof_pos[:]
        self.dof_vel[:, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[:]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.refresh_all_tensors()

    def set_friction_damping_params(self, joint_friction=None, joint_damping=None):
        """
        Set friction and damping parameters for the robot joints.

        Args:
            joint_friction: Friction values for the joints.
            joint_damping: Damping values for the joints.
        """
        if joint_friction is None and joint_damping is None:
            return

        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)

            if joint_friction is not None:
                franka_dof_props['friction'][:9] = joint_friction[:9]
            if joint_damping is not None:
                franka_dof_props['damping'][:9] = joint_damping[:9]

            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

    def get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # Plug-socket tensors
        self.plug_keypoint_origin_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.socket_keypoint_origin_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat((self.num_envs, 1))
        insertion_height = (1 - self.cfg_task.rl.insertion_frac)
        self.socket_keypoint_origin_local[:, 2] = self.socket_heights.squeeze(-1)[:] * insertion_height
        self.socket_tip_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat((self.num_envs, 1))
        self.socket_tip_pos_local[:, 2] = self.socket_heights.squeeze(-1)[:]

        # Keypoint tensors
        self.keypoint_offsets = \
            self.get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                          dtype=torch.float32,
                                          device=self.device)
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

        self.identity_quat = \
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)

        self._actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        # Compute pos of keypoints on gripper, plug, and socket in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_quat,
                                                                     self.plug_pos,
                                                                     self.identity_quat,
                                                                     (keypoint_offset + self.plug_keypoint_origin_local))[1]
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_quat,
                                                                       self.socket_pos,
                                                                       self.identity_quat,
                                                                       (keypoint_offset + self.socket_keypoint_origin_local))[1]

    def refresh_all_tensors(self):
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

    def pre_physics_step(self, actions):
        """Optionally reset environments at the end of episodes. Apply actions from policy as position/rotation targets."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self._actions,
                                            ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_close_gripper_width", 0.0),
                                            do_scale=True)

        sim_dt_noise = self.cfg_task.env.get("sim_dt_noise", 0)
        if sim_dt_noise > 0.0:
            sim_params = self.gym.get_sim_params(self.sim)
            sim_params.dt = self.cfg_base.sim.dt * (1 + torch.rand(1) * sim_dt_noise)
            self.gym.set_sim_params(self.sim, sim_params)

        num_extra_control_steps = self.cfg_task.env.get("num_additional_control_steps", 0)  # for backward compatibility
        num_additional_control_steps_noise = self.cfg_task.env.get("num_additional_control_steps_noise", 0)
        if num_additional_control_steps_noise:
            num_extra_control_steps += torch.randint(0,
                                                     self.cfg_task.env.num_additional_control_steps_noise + 1,
                                                     (1,)
                                                     )[0].item()
        self.execute_control_loop(num_extra_control_steps)

    def execute_control_loop(self, num_control_steps):
        """
        Execute the control loop for a specified number of steps.

        Args:
            num_control_steps: Number of control steps to execute.
        """
        for _ in range(num_control_steps):
            # execute previous control signal
            self.gym.simulate(self.sim)

            # refresh tensors
            self.refresh_all_tensors()

            # generate new control signal
            self.generate_ctrl_signals()

    def apply_screw_primitive_task_space(self):
        """Apply screw primitive in task space."""
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()

        # get screw motion as delta euler angle
        screw_motion = torch_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([0.0], device=self.device),
            torch.tensor([np.pi/2], device=self.device)).expand(self.num_envs, 4)
        self.ctrl_target_fingertip_midpoint_pos[:, 2] -= 0.005  # move down while screwing
        self.ctrl_target_fingertip_midpoint_quat[:] = torch_utils.quat_mul(self.fingertip_midpoint_quat,
                                                                           screw_motion)

        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos=0.0, gentle_gripper_close=False,
                                                    sim_steps=self.cfg_task.env.num_gripper_close_sim_steps * 2)

    def apply_screw_primitive(self):
        """Apply screw primitive."""
        self.refresh_all_tensors()
        target_dof = self.dof_pos.clone()
        target_dof[:, 6] += 1.

        DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
        DEFAULT_D_GAINS = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
        joint_prop_gains = torch.tensor(DEFAULT_K_GAINS, device=self.device) / 100.
        joint_deriv_gains = torch.tensor(DEFAULT_D_GAINS, device=self.device) / 100.

        # Step sim
        sim_steps = self.cfg_task.env.num_gripper_close_sim_steps * 4
        for _ in range(sim_steps):
            self.refresh_all_tensors()
            self.dof_torque[:, 0:7] = joint_prop_gains * (target_dof - self.dof_pos)[:, 0:7] + \
                                      joint_deriv_gains * (0.0 - self.dof_vel[:, 0:7])
            # keep prev torque applied by gripper
            self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.dof_torque),
                                                            gymtorch.unwrap_tensor(self.actor_ids_sim_tensors['franka']),
                                                            len(self.actor_ids_sim_tensors['franka']))
            self.render()
            self.gym.simulate(self.sim)

    def execute_terminal_primitive(self):
        """Execute terminal primitive actions."""
        if self.cfg_task.env.task_type == 'screwing':
            # do screw primitive
            self.apply_screw_primitive()

        if self.cfg_task.env.task_type in ['placement', 'screwing']:
            # open-gripper and lift
            self._open_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps//2)
            self._lift_gripper(gripper_dof_pos=0.1, sim_steps=self.cfg_task.env.num_gripper_close_sim_steps//2)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            self.execute_terminal_primitive()

        self.refresh_all_tensors()
        self.compute_observations()
        self.compute_reward()

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            # Check if plug is at the goal location within the socket
            task_success = self._check_success()
            self.rew_buf[:] += task_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(task_success.float())

        self.prev_actions[:] = self._actions

        if self.cfg_base.mode.export_states:
            self.extract_poses()

    def compute_observations(self):
        """Compute observations."""

        if self.cfg_task.env.use_dict_obs:
            return self.compute_observations_dict_obs()

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_observations_dict_obs(self):
        """
        Compute observations as a dictionary.

        Returns:
            obs_dict: Dictionary containing observations.
        """
        self.obs_dict['ee_pos'][:] = self.fingertip_midpoint_pos
        self.obs_dict['ee_quat'][:] = self.fingertip_midpoint_quat
        self.obs_dict['socket_pos'][:] = self.socket_pos
        self.obs_dict['socket_quat'][:] = self.socket_quat
        if 'plug_pos' in self.cfg_task.env.obsDims or self.cfg_task.rl.asymmetric_observations:
            self.obs_dict['plug_pos'][:] = self.plug_pos
            self.obs_dict['plug_quat'][:] = self.plug_quat
        if 'eef_to_plug_pos' in self.cfg_task.env.obsDims:
            eef_to_plug_transform = torch_jit_utils.tf_combine(
                *torch_jit_utils.tf_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_pos),
                self.plug_quat, self.plug_pos
            )
            self.obs_dict['eef_to_plug_pos'][:] = eef_to_plug_transform[1]
            self.obs_dict['eef_to_plug_quat'][:] = eef_to_plug_transform[0]
        if 'dof_pos' in self.cfg_task.env.obsDims or 'dof_pos' in self.cfg_task.env.stateDims:
            self.obs_dict['dof_pos'][:] = self.dof_pos
        if 'dof_vel' in self.cfg_task.env.obsDims or 'dof_vel' in self.cfg_task.env.stateDims:
            self.obs_dict['dof_vel'][:] = self.dof_vel
        if 'ee_lin_vel' in self.cfg_task.env.obsDims or self.cfg_task.rl.asymmetric_observations:
            self.obs_dict['ee_lin_vel'][:] = self.fingertip_midpoint_linvel
            self.obs_dict['ee_ang_vel'][:] = self.fingertip_midpoint_angvel
        if self.cfg_task.rl.add_contact_force_plug_decomposed or self.cfg_task.rl.add_contact_info_to_aac_states:
            self.obs_dict['plug_socket_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.socket_body_id_env]
            if self.cfg_task.env.use_compliant_contact:
                self.obs_dict['plug_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.franka_body_ids_env['elastomer_left']]
                self.obs_dict['plug_right_elastomer_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.franka_body_ids_env['elastomer_right']]
            else:
                self.obs_dict['plug_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.franka_body_ids_env['panda_leftfinger']]
                self.obs_dict['plug_right_elastomer_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.franka_body_ids_env['panda_rightfinger']]
        if self.cfg_task.env.use_camera_obs:
            images = self.get_camera_image_tensors_dict()

            if self.cfg_task.env.use_isaac_gym_tactile:
                # Optionally subsample tactile image
                ssr = self.cfg_task.env.tactile_subsample_ratio
                for k in self.tactile_ig_keys:
                    images[k] = images[k][:, ::ssr, ::ssr]

            for cam in images:
                if cam in self.cfg_task.env.obsDims:
                    if images[cam].dtype == torch.uint8:
                        self.obs_dict[cam][..., :3] = images[cam] / 255.
                    else:
                        self.obs_dict[cam][..., :3] = images[cam]

            self.apply_image_augmentation_to_obs_dict()

        self.obs_dict['socket_pos'][:] = self.socket_pos + self.socket_obs_noise
        self.obs_dict['socket_pos_gt'][:] = self.socket_pos

        if self.cfg_task.env.use_shear_force:
            tactile_force_field_dict = self.get_tactile_force_field_tensors_dict()
            if self.cfg_task.env.use_tactile_field_obs:
                for k in ['tactile_force_field_left', 'tactile_force_field_right']:
                    self.obs_dict[k][:] = tactile_force_field_dict[k]
                    if self.cfg_task.env.zero_out_normal_force_field_obs:
                        self.obs_dict[k][..., 0] *= 0.0
        return self.obs_dict

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if episode length expired."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at the current timestep."""

        keypoint_diff = self.keypoints_socket - self.keypoints_plug
        keypoint_dist = torch.mean(torch.norm(keypoint_diff, p=2, dim=-1), dim=-1)
        keypoint_reward = -keypoint_dist

        uncentered_plug_dist_below_socket = self._get_peg_tip_distance_when_not_centered()

        # a, b = 50, 2
        a, b = 300, 0.0001
        a, b = 50, 0.0001
        keypoint_reward_exp = 1. / (torch.exp(a * keypoint_reward) + b + torch.exp(-a * keypoint_reward))
        if self.cfg_task.rl.use_shaped_keypoint_reward:
            # keypoint_reward_exp[uncentered_plug_dist_below_socket > 0] *= -1.  # penalize if peg is beside socket
            keypoint_reward[uncentered_plug_dist_below_socket > 0] *= 10.  # penalize if peg is beside socket
        action_penalty = torch.norm(self._actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self._actions - self.prev_actions, p=2, dim=-1)
        contact_penalty = torch.norm(self.contact_force_pairwise[:, self.socket_body_id_env], p=2, dim=-1).sum(1)
        contact_force_table = torch.norm(self.contact_force_pairwise[:, self.table_body_id], p=2, dim=-1).sum(1)
        plug_socket_force = self.contact_force_pairwise[:, self.plug_body_id_env, self.socket_body_id_env].clone()
        contact_force_plug_socket = torch.norm(plug_socket_force, p=1, dim=-1)
        finger_body_ids = [self.franka_body_ids_env['panda_leftfinger'], self.franka_body_ids_env['panda_rightfinger']]
        socket_fingers_force = self.contact_force_pairwise[:, self.socket_body_id_env, finger_body_ids].clone()
        contact_force_socket_fingers = torch.norm(socket_fingers_force, p=2, dim=-1).sum(1)

        if self.cfg_task.rl.use_shaped_contact_pen:
            # allow more contact interactions at the socket hole
            plug_tip_pos = self.plug_pos
            _, socket_tip_pos = torch_jit_utils.tf_combine(self.socket_quat, self.socket_pos,
                                                           self.identity_quat, self.socket_tip_pos_local)
            contact_rich_region_size_factor = self.cfg_task.rl.contact_rich_region_size_factor  # 0.1
            planar_threshold = self.socket_diameters * contact_rich_region_size_factor
            # is_tip_centre_on_path
            is_plug_centered = torch.norm((plug_tip_pos - socket_tip_pos)[:, :2], dim=-1) < planar_threshold.squeeze(-1)

            # if is_plug_centered, reduce the contact penalty by a scale factor
            contact_pen_reduction_scalar = self.cfg_task.rl.contact_pen_reduction_scalar  # = 0.001
            contact_penalty = (contact_penalty * (1 - is_plug_centered.float()) +
                               contact_penalty * (is_plug_centered.float() * contact_pen_reduction_scalar))

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          + keypoint_reward_exp * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          - action_grad_penalty * self.cfg_task.rl.action_gradient_penalty_scale \
                          - contact_force_table * self.cfg_task.rl.contact_penalty_scale \
                          - contact_penalty * self.cfg_task.rl.contact_penalty_scale

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.cfg_task.randomize.randomize_compliance:
            if self.cfg_task.env.use_compliant_contact:
                # sample_compliance
                k_range = self.cfg_task.randomize.compliance_stiffness_range
                d_range = self.cfg_task.randomize.compliance_damping_range
                ks = k_range[0] + torch.rand(self.num_envs, device=self.device) * (k_range[1] - k_range[0])
                ds = d_range[0] + torch.rand(self.num_envs, device=self.device) * (d_range[1] - d_range[0])
                # set sampled compliance params for each env
                for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
                    self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                                      elastomer_link_name=elastomer_link_name,
                                                      compliance_stiffness=ks,
                                                      compliant_damping=ds,
                                                      use_acceleration_spring=False)

        if self.cfg_task.randomize.randomize_ctrl_params:
            assert self.cfg_task.ctrl.ctrl_type == 'task_space_impedance', 'controller randomization currently works only for task_space_impedance'
            # use default controller params when randomizing initial state
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))

        if self.cfg_task.ige_dr.randomize:
            # use initial joint friction and damping during environment initialization
            for env_id in range(self.num_envs):
                env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']
                franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
                franka_dof_props['friction'][7:9] = self.cfg_task.env.default_gripper_joint_friction
                franka_dof_props['damping'][7:9] = self.cfg_task.env.default_gripper_joint_damping
                self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self.disable_gravity()  # to prevent plug from falling
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()

        self._move_gripper_to_dof_pos(gripper_dof_pos=self.cfg_task.env.get("franka_close_gripper_width", 0.0),
                                      sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        self.enable_gravity(gravity_vec=self.cfg_base.sim.gravity)

        if self.cfg_task.randomize.randomize_ctrl_params:
            self.randomize_controller_params()

        if 'ige_dr' in self.cfg_task and self.cfg_task.ige_dr.randomize:
            # Must be executed before resetting self.reset_buf
            self.envs = self.env_ptrs  # DR code looks for self.envs
            self.apply_randomizations(self.cfg_task.ige_dr.randomization_params)
            self.set_gripper_friction_to_default()  # Don't randomize gripper friction/dynamic params, reset to default values

        self._reset_buffers(env_ids)

        unit_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        self.socket_obs_noise = unit_noise * torch.tensor(self.cfg_task.randomize.socket_pos_xyz_obs_noise,
                                                          device=self.device).expand(self.num_envs, 3)
        self.reset_image_augmentation()

    def set_gripper_friction_to_default(self):
        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
            franka_dof_props['friction'][7:9] = self.cfg_task.env.default_gripper_joint_friction
            franka_dof_props['damping'][7:9] = self.cfg_task.env.default_gripper_joint_damping
            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

    def randomize_controller_params(self):

        k_gains_min = torch.tensor(self.cfg_task.randomize.task_prop_gains_min, dtype=torch.float32, device=self.device)
        k_gains_max = torch.tensor(self.cfg_task.randomize.task_prop_gains_max, dtype=torch.float32, device=self.device)
        ctrl_param_noise = torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device)

        self.cfg_ctrl['task_prop_gains'] = k_gains_min + ctrl_param_noise * (k_gains_max - k_gains_min)
        self.cfg_ctrl['task_deriv_gains'] = 2 * torch.sqrt(self.cfg_ctrl['task_prop_gains'])

        # Scale down the rotation gains of the controller. Stabilizes motion for the default franka urdf.
        self.cfg_ctrl['task_deriv_gains'][:, 3:] /= 10.     # reduce the damping of the rotation action params

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka to initial joint configuration
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))

        self._reset_franka_actuation(self.ctrl_target_dof_pos)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps,
                                     ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_open_gripper_width",
                                                                                       self.asset_info_franka_table.franka_gripper_width_max))

    def _reset_franka_actuation(self, ctrl_target_dof_pos):
        multi_env_ids_int32 = self.actor_ids_sim_tensors['franka'].flatten()
        zeros = torch.zeros_like(self.dof_torque)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(zeros),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

    def _reset_object(self, env_ids):
        """Reset root states of plug and socket."""
        self.gym.simulate(self.sim)
        self.refresh_base_tensors()

        # Move peg into hand. How much of peg is in hand?
        ee_to_plug_tip_pos_local = torch.zeros_like(self.fingertip_midpoint_pos)
        min_tip_dist = self.socket_heights.squeeze(-1) * 1.1  # required to ensure task is solvable i.e. peg can go full into the socket
        peg_in_hand_buffer = self.cfg_task.env.peg_in_hand_buffer
        min_tip_dist *= peg_in_hand_buffer
        max_tip_dist = self.plug_lengths.squeeze(-1) * 0.8  # required to ensure peg is within the gripper
        # when min_tip_dist, plug is fully in the hand with the tip aligned with the finger tip with just enough to go into the socket
        # when max_tip_dist, plug is fully out the hand with the tip aligned with the finger tip, barely touching the fingertip to stay in hand
        tip_dist_range_center = (min_tip_dist + max_tip_dist) / 2.
        tip_dist_range_mag = max_tip_dist - min_tip_dist
        plug_pos_in_gripper_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        plug_pos_in_gripper_z_sampled = tip_dist_range_center + \
                                        (plug_pos_in_gripper_noise[:, 2] *
                                         tip_dist_range_mag / 2. *
                                         self.cfg_task.randomize.plug_pos_z_in_gripper_noise_multiplier)
        # subtract from plug length to get the distance from the tip
        ee_to_plug_tip_pos_local[:, 2] = plug_pos_in_gripper_z_sampled - self.plug_lengths.squeeze(-1)

        plug_pos_in_gripper_xy_sampled = plug_pos_in_gripper_noise[:, :2] @ torch.diag(
            torch.tensor(self.cfg_task.randomize.plug_pos_in_gripper_noise_xy, device=self.device))
        ee_to_plug_tip_pos_local[:, :2] = plug_pos_in_gripper_xy_sampled

        world_to_plug_tip_quat, world_to_plug_tip_pos = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                                   self.fingertip_midpoint_pos,
                                                                                   self.identity_quat,
                                                                                   ee_to_plug_tip_pos_local)

        # Rotate peg in hand
        plug_noise_rot_in_gripper = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        plug_noise_rot_in_gripper *= torch.tensor(self.cfg_task.randomize.plug_noise_rot_in_gripper,
                                                  device=self.device).expand(self.num_envs, 3)
        # z is along the axis of the gripper so it has no effects for round pegs
        # y rotates in and out of the gripper axis
        # x rot should be zero
        zero_translation = torch.zeros_like(world_to_plug_tip_pos)
        ee_to_plug_tip_rot_quat = torch_utils.quat_from_euler_xyz(plug_noise_rot_in_gripper[:, 0],
                                                                  plug_noise_rot_in_gripper[:, 1],
                                                                  plug_noise_rot_in_gripper[:, 2])

        world_to_plug_tip_quat, world_to_plug_tip_pos = torch_jit_utils.tf_combine(world_to_plug_tip_quat,
                                                                                   world_to_plug_tip_pos,
                                                                                   ee_to_plug_tip_rot_quat,
                                                                                   zero_translation)

        # Translate from peg tip in hand to peg base
        plug_tip_to_base_local = torch.zeros_like(self.fingertip_midpoint_pos)
        plug_tip_to_base_local[:, 2] = self.plug_lengths.squeeze()
        world_to_plug_base_quat, world_to_plug_base_pos = torch_jit_utils.tf_combine(world_to_plug_tip_quat,
                                                                                     world_to_plug_tip_pos,
                                                                                     self.identity_quat,
                                                                                     plug_tip_to_base_local)

        # flip orientation
        flip_z_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        world_to_plug_base_quat, world_to_plug_base_pos = torch_jit_utils.tf_combine(world_to_plug_base_quat,
                                                                                     world_to_plug_base_pos,
                                                                                     flip_z_quat,
                                                                                     zero_translation)
        self.root_pos[env_ids, self.plug_actor_id_env, :] = world_to_plug_base_pos
        self.root_quat[env_ids, self.plug_actor_id_env] = world_to_plug_base_quat

        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        # Randomize root state of socket
        socket_noise_xyz = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        socket_noise_xyz = socket_noise_xyz @ torch.diag(
            torch.tensor(self.cfg_task.randomize.socket_pos_xyz_noise, dtype=torch.float32, device=self.device))
        self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.cfg_task.randomize.socket_pos_xyz_initial[0] + \
                                                            socket_noise_xyz[env_ids, 0]
        self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.cfg_task.randomize.socket_pos_xyz_initial[1] + \
                                                            socket_noise_xyz[env_ids, 1]
        self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_task.randomize.socket_pos_xyz_initial[2] + \
                                                            socket_noise_xyz[env_ids, 2]

        socket_rot_initial = self.cfg_task.randomize.socket_rot_initial
        socket_rot_noise_level = self.cfg_task.randomize.socket_rot_noise

        socket_rot_euler = torch.tensor(socket_rot_initial, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        socket_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        socket_rot_noise = socket_rot_noise @ torch.diag(torch.tensor(socket_rot_noise_level, device=self.device))

        socket_rot_euler += socket_rot_noise
        self.root_quat[env_ids, self.socket_actor_id_env] = torch_utils.quat_from_euler_xyz(
            socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2])

        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        plug_socket_actor_ids_sim = torch.cat((self.actor_ids_sim_tensors['plug'][env_ids],
                                               self.actor_ids_sim_tensors['socket'][env_ids]),
                                              dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_socket_actor_ids_sim),
                                                     len(plug_socket_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-0.632, -0.221,  0.7196)
        cam_target = gymapi.Vec3(0., 0.4, 0.58)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """
        Apply actions from policy as position/rotation targets.

        Args:
            actions: Actions to be applied.
            ctrl_target_gripper_dof_pos: Target gripper DOF position.
            do_scale: Boolean indicating if action scaling is applied.
        """
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        # Keep current end-effector pose as target end-effector pose, when moving the gripper joint
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()

        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos, sim_steps=sim_steps)

    def _lift_gripper(self, gripper_dof_pos=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += lift_distance
        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos, sim_steps=sim_steps)

    def _move_to_target_pose_and_gripper_width(self, target_fingertip_midpoint_pos, target_fingertip_midpoint_quat,
                                               gripper_dof_pos, sim_steps=20, gentle_gripper_close=False):
        """Move arm to target end-effector pose, and gripper to target width using task-space controller."""

        # Keep current end-effector pose as target end-effector pose, when moving the gripper joint
        self.ctrl_target_fingertip_midpoint_pos[:] = target_fingertip_midpoint_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = target_fingertip_midpoint_quat

        # Step sim
        for step_i in range(sim_steps):
            self.refresh_all_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            if gentle_gripper_close:
                # break gripper motion into steps. Try to move to target in half the num of steps left.
                num_steps_left = sim_steps - step_i
                target_gripper_i = (gripper_dof_pos - self.gripper_dof_pos) / (0.5 * num_steps_left)
                target_gripper_pos = self.gripper_dof_pos + target_gripper_i
            else:
                target_gripper_pos = gripper_dof_pos
            self._apply_actions_as_ctrl_targets(delta_hand_pose, target_gripper_pos, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)

    def _get_peg_tip_distance_when_not_centered(self):
        """Get distance of the tip of the peg below the level of the socket, when the peg is not centered.
        Used to penalize a common failure case when the peg is upright beside the socket and not inside the socket
        """
        plug_tip_pos = self.plug_pos
        _, socket_tip_pos = torch_jit_utils.tf_combine(self.socket_quat, self.socket_pos,
                                                       self.identity_quat, self.socket_tip_pos_local)
        planar_threshold = self.socket_diameters * 0.5
        # is_tip_centre_on_path
        is_plug_centered = torch.norm((plug_tip_pos - socket_tip_pos)[:, :2], dim=-1) < planar_threshold.squeeze(-1)
        # is_tip_below_socket_opening
        plug_dist_below_socket = (plug_tip_pos - socket_tip_pos)[:, 2]
        plug_dist_below_socket[plug_dist_below_socket > 0] = 0  # zero out points above the socket
        plug_dist_below_socket = -plug_dist_below_socket    # distance magnitude below socket
        not_centered_dist_below_socket_tip = (1.0 - is_plug_centered.float()) * plug_dist_below_socket

        return not_centered_dist_below_socket_tip

    def _check_plug_close_to_socket(self):
        """Check if plug is close to socket."""

        keypoint_dist = torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1)

        is_plug_close_to_socket = torch.where(torch.mean(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                              torch.ones_like(self.progress_buf),
                                              torch.zeros_like(self.progress_buf))

        return is_plug_close_to_socket

    def _check_plug_is_centered_on_socket(self):
        """Get distance of the tip of the peg below the level of the socket, when the peg is not centered.
        Used to penalize a common failure case when the peg is upright beside the socket and not inside the socket
        """
        plug_tip_pos = self.plug_pos
        _, socket_tip_pos = torch_jit_utils.tf_combine(self.socket_quat, self.socket_pos,
                                                       self.identity_quat, self.socket_tip_pos_local)
        planar_threshold = self.socket_diameters * 0.6
        is_plug_centered = torch.norm((plug_tip_pos - socket_tip_pos)[:, :2], dim=-1) < planar_threshold.squeeze(-1)

        return is_plug_centered

    def _check_plug_is_upright(self):
        """Check if plug is upright/aligned with socket."""
        principal_axis = [0., 0., 1.]
        principal_axis_tensor = torch.tensor(principal_axis, device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        plug_principal_axis = torch_utils.quat_rotate(self.plug_quat, principal_axis_tensor)
        socket_principal_axis = torch_utils.quat_rotate(self.socket_quat, principal_axis_tensor)
        cos_angle_between_principal_axis = torch.sum(plug_principal_axis * socket_principal_axis, axis=-1)

        upright_angle_threshold = 15    # degrees
        upright_angle_threshold = np.deg2rad(upright_angle_threshold)

        is_plug_upright = cos_angle_between_principal_axis > np.cos(upright_angle_threshold)
        return is_plug_upright

    def _check_success(self):
        """Check for task success."""

        if self.cfg_task.env.task_type == 'insertion':
            task_success = self._check_plug_close_to_socket()
        elif self.cfg_task.env.task_type == 'placement':
            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
            # This additional conditional is to ensure that the final step primitive has been executed.
            # Useful when calling the success check outside of RL environment TODO: Find a good location for logic.
            if is_last_step:
                is_upright = self._check_plug_is_upright()
                is_centered = self._check_plug_is_centered_on_socket()
                task_success = torch.logical_and(is_upright, is_centered)
            else:
                task_success = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)
        else:
            is_upright = self._check_plug_is_upright()
            is_centered = self._check_plug_is_centered_on_socket()
            task_success = torch.logical_and(is_upright, is_centered)
        return task_success

    def _randomize_gripper_pose(self, env_ids, sim_steps, ctrl_target_gripper_dof_pos=0.0):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial,
                                                               device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
            self.num_envs, 1)

        fingertip_midpoint_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
                                                do_scale=False)

            self.gym.simulate(self.sim)
            self.refresh_all_tensors()
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.actor_ids_sim_tensors['franka'][env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self._reset_franka_actuation(self.dof_pos.clone())

        self.refresh_all_tensors()
