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

"""TacSL: base class.

Inherits Factory base class and Factory abstract base class. Inherited by TacSL environment classes. Not directly executed.

Configuration defined in TacSLBase.yaml. Asset info defined in tacsl_asset_info_franka_table.yaml.
"""


import hydra
import math
import numpy as np
import os
import sys
import torch

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.factory.factory_base import FactoryBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_base import FactoryABCBase
from isaacgymenvs.tasks.factory.factory_schema_config_base import FactorySchemaConfigBase


class TacSLBase(FactoryBase, FactoryABCBase):

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_base', node=FactorySchemaConfigBase)

        config_path = 'task/TacSLBase.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base['task']  # strip superfluous nesting

        asset_info_path = '../../assets/tacsl/yaml/tacsl_asset_info_franka_table.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table['']['']['']['']['']['']['assets']['tacsl']['yaml']  # strip superfluous nesting

    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = self.asset_info_franka_table.table_height  # default = 0.0
        plane_params.static_friction = 1.0  # default = 1.0
        plane_params.dynamic_friction = 1.0  # default = 1.0
        plane_params.restitution = 0.0  # default = 0.0

        self.gym.add_ground(self.sim, plane_params)

    def import_franka_assets(self):
        """Set Franka and table asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets',
                                 self.cfg_task.env.franka_asset_dir, 'urdf')

        franka_options = gymapi.AssetOptions()
        franka_options.flip_visual_attachments = True
        franka_options.fix_base_link = True
        franka_options.collapse_fixed_joints = False
        franka_options.thickness = 0.0  # default = 0.02
        franka_options.density = 1000.0  # default = 1000.0
        franka_options.armature = 0.01  # default = 0.0
        franka_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            franka_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
            franka_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
            franka_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
            franka_options.max_angular_velocity = 2 * math.pi  # default = 64.0; reduced to prevent CUDA errors
        else:
            franka_options.linear_damping = 0.0  # default = 0.0
            franka_options.max_linear_velocity = 1000.0  # default = 1000.0
            franka_options.angular_damping = 0.5  # default = 0.5
            franka_options.max_angular_velocity = 64.0  # default = 64.0
        franka_options.disable_gravity = True
        franka_options.enable_gyroscopic_forces = True
        franka_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        franka_options.use_mesh_materials = True
        if self.cfg_base.mode.export_scene:
            franka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        table_options = gymapi.AssetOptions()
        table_options.flip_visual_attachments = False  # default = False
        table_options.fix_base_link = True
        table_options.thickness = 0.0  # default = 0.02
        table_options.density = 1000.0  # default = 1000.0
        table_options.armature = 0.0  # default = 0.0
        table_options.use_physx_armature = True
        table_options.linear_damping = 0.0  # default = 0.0
        table_options.max_linear_velocity = 1000.0  # default = 1000.0
        table_options.angular_damping = 0.0  # default = 0.5
        table_options.max_angular_velocity = 64.0  # default = 64.0
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        franka_asset_file_path = os.path.join(os.path.abspath(urdf_root), self.cfg_task.env.franka_urdf_file)
        self.asset_file_paths['franka'] = franka_asset_file_path
        franka_asset = self.gym.load_asset(self.sim, urdf_root, self.cfg_task.env.franka_urdf_file, franka_options)
        table_asset = self.gym.create_box(self.sim,
                                          self.asset_info_franka_table.table_depth,
                                          self.asset_info_franka_table.table_width,
                                          self.asset_info_franka_table.table_height,
                                          table_options)

        return franka_asset, table_asset

    def acquire_base_tensors(self):
        """Acquire and wrap tensors. Create views."""

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)  # shape = (num_envs * num_dofs, 1)
        _contact_force_net = self.gym.acquire_net_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies, 3)
        _contact_force_pairwise = self.gym.acquire_pairwise_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies * num_bodies, 3)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'franka')  # shape = (num envs, num_bodies, 6, num_dofs)
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'franka')  # shape = (num_envs, num_dofs, num_dofs)

        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.dof_force = gymtorch.wrap_tensor(_dof_force)
        self.contact_force_net = gymtorch.wrap_tensor(_contact_force_net)
        self.contact_force_pairwise = gymtorch.wrap_tensor(_contact_force_pairwise)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 0:3]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 10:13]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_force_view = self.dof_force.view(self.num_envs, self.num_dofs, 1)[..., 0]
        self.contact_force_net = self.contact_force_net.view(self.num_envs, self.num_bodies, 3)[..., 0:3]
        if not self.cfg_base.mode.export_scene:
            # export scene always uses CPU, but pairwise contact only exist for GPU
            self.contact_force_pairwise = self.contact_force_pairwise.view(self.num_envs, self.num_bodies, self.num_bodies, 3)[..., 0:3]

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_dof_vel = self.dof_vel[:, 0:7]
        self.arm_mass_matrix = self.mass_matrix[:, 0:7, 0:7]  # for Franka arm (not gripper)

        self.hand_pos = self.body_pos[:, self.hand_body_id_env, 0:3]
        self.hand_quat = self.body_quat[:, self.hand_body_id_env, 0:4]
        self.hand_linvel = self.body_linvel[:, self.hand_body_id_env, 0:3]
        self.hand_angvel = self.body_angvel[:, self.hand_body_id_env, 0:3]
        self.hand_jacobian = self.jacobian[:, self.hand_body_id_env_actor - 1, 0:6, 0:7]  # minus 1 because base is fixed

        self.left_finger_pos = self.body_pos[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_quat = self.body_quat[:, self.left_finger_body_id_env, 0:4]
        self.left_finger_linvel = self.body_linvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_angvel = self.body_angvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_jacobian = self.jacobian[:, self.left_finger_body_id_env_actor - 1, 0:6, 0:7]  # minus 1 because base is fixed

        self.right_finger_pos = self.body_pos[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_quat = self.body_quat[:, self.right_finger_body_id_env, 0:4]
        self.right_finger_linvel = self.body_linvel[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_angvel = self.body_angvel[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_jacobian = self.jacobian[:, self.right_finger_body_id_env_actor - 1, 0:6, 0:7]  # minus 1 because base is fixed

        self.left_fingertip_pos = self.body_pos[:, self.left_fingertip_body_id_env, 0:3]
        self.left_fingertip_quat = self.body_quat[:, self.left_fingertip_body_id_env, 0:4]
        self.left_fingertip_linvel = self.body_linvel[:, self.left_fingertip_body_id_env, 0:3]
        self.left_fingertip_angvel = self.body_angvel[:, self.left_fingertip_body_id_env, 0:3]
        self.left_fingertip_jacobian = self.jacobian[:, self.left_fingertip_body_id_env_actor - 1, 0:6, 0:7]  # minus 1 because base is fixed

        self.right_fingertip_pos = self.body_pos[:, self.right_fingertip_body_id_env, 0:3]
        self.right_fingertip_quat = self.body_quat[:, self.right_fingertip_body_id_env, 0:4]
        self.right_fingertip_linvel = self.body_linvel[:, self.right_fingertip_body_id_env, 0:3]
        self.right_fingertip_angvel = self.body_angvel[:, self.right_fingertip_body_id_env, 0:3]
        self.right_fingertip_jacobian = self.jacobian[:, self.right_fingertip_body_id_env_actor - 1, 0:6, 0:7]  # minus 1 because base is fixed

        if not self.cfg_base.mode.export_scene:
            self.left_finger_force = self.contact_force_pairwise[:, self.left_finger_body_id_env].sum(1)
            self.right_finger_force = self.contact_force_pairwise[:, self.right_finger_body_id_env].sum(1)

        self.gripper_dof_pos = self.dof_pos[:, 7:9]

        self.fingertip_centered_pos = self.body_pos[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_quat = self.body_quat[:, self.fingertip_centered_body_id_env, 0:4]
        self.fingertip_centered_linvel = self.body_linvel[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_angvel = self.body_angvel[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_jacobian = self.jacobian[:, self.fingertip_centered_body_id_env_actor - 1, 0:6, 0:7]  # minus 1 because base is fixed

        self.fingertip_midpoint_pos = self.fingertip_centered_pos.detach().clone()  # initial value
        self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal
        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel.detach().clone()  # initial value
        # From sum of angular velocities (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand; angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.
        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5  # approximation

        self.dof_torque = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.ctrl_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.ctrl_target_gripper_dof_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.ctrl_target_fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

    def refresh_base_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_pairwise_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos) * 0.5
        self.fingertip_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_franka_table.franka_finger_length,
                                                                 device=self.device)

        if not self.cfg_base.mode.export_scene:
            self.left_finger_force = self.contact_force_pairwise[:, self.left_finger_body_id_env].sum(1)
            self.right_finger_force = self.contact_force_pairwise[:, self.right_finger_body_id_env].sum(1)
        self.fingertip_midpoint_pos = (self.left_fingertip_pos + self.right_fingertip_pos) * 0.5
        # TODO: Add relative velocity term (see https://dynamicsmotioncontrol487379916.files.wordpress.com/2020/11/21-me258pointmovingrigidbody.pdf)
        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(self.fingertip_centered_angvel,
                                                                                      (self.fingertip_midpoint_pos - self.fingertip_centered_pos),
                                                                                      dim=1)
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5  # approximation

    def parse_controller_spec(self):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {'num_envs',
                         'jacobian_type',
                         'gripper_prop_gains',
                         'gripper_deriv_gains',
                         'motor_ctrl_mode',
                         'gain_space',
                         'ik_method',
                         'joint_prop_gains',
                         'joint_deriv_gains',
                         'do_motion_ctrl',
                         'task_prop_gains',
                         'task_deriv_gains',
                         'do_inertial_comp',
                         'motion_ctrl_axes',
                         'do_force_ctrl',
                         'force_ctrl_method',
                         'wrench_prop_gains',
                         'force_ctrl_axes'}
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl['num_envs'] = self.num_envs
        self.cfg_ctrl['jacobian_type'] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_prop_gains,
                                                           device=self.device).repeat((self.num_envs, 1))
        self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_deriv_gains,
                                                            device=self.device).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == 'gym_default':
            self.cfg_ctrl['motor_ctrl_mode'] = 'gym'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.gripper_prop_gains,
                                                               device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.gripper_deriv_gains,
                                                                device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'joint_space_ik':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
        elif ctrl_type == 'joint_space_id':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
        elif ctrl_type == 'task_space_impedance':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
            self.cfg_ctrl['do_null_space_ctrl'] = False
        elif ctrl_type == 'task_space_nullspace':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_nullspace.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_nullspace.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.task_space_nullspace.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
            self.cfg_ctrl['do_null_space_ctrl'] = True
            self.cfg_ctrl['kp_null'] = self.cfg_task.ctrl.task_space_nullspace.kp_null
            self.cfg_ctrl['kd_null'] = self.cfg_task.ctrl.task_space_nullspace.kd_null
            self.cfg_ctrl['default_dof_pos_tensor'] = torch.tensor(self.cfg_task.ctrl.task_space_nullspace.default_dof_pos_tensor,
                                                                   device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'operational_space_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
            self.cfg_ctrl['do_null_space_ctrl'] = False
        elif ctrl_type == 'open_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'open'
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.open_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_null_space_ctrl'] = False
        elif ctrl_type == 'closed_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_null_space_ctrl'] = False
        elif ctrl_type == 'hybrid_force_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_null_space_ctrl'] = False

        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            prop_gains = torch.cat((self.cfg_ctrl['joint_prop_gains'],
                                    self.cfg_ctrl['gripper_prop_gains']), dim=-1).to('cpu')
            deriv_gains = torch.cat((self.cfg_ctrl['joint_deriv_gains'],
                                     self.cfg_ctrl['gripper_deriv_gains']), dim=-1).to('cpu')
            # No tensor API for getting/setting actor DOF props; thus, loop required
            franka_handle = self.actor_handles['franka']
            for env_ptr, prop_gain, deriv_gain in zip(self.env_ptrs, prop_gains, deriv_gains):
                franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
                franka_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
                franka_dof_props['stiffness'] = prop_gain
                franka_dof_props['damping'] = deriv_gain
                self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            # No tensor API for getting/setting actor DOF props; thus, loop required
            franka_handle = self.actor_handles['franka']
            for env_ptr in self.env_ptrs:
                franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
                franka_dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
                franka_dof_props['stiffness'][:] = 0.0  # zero passive stiffness
                franka_dof_props['damping'][:] = 0.0  # zero passive damping
                self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

        if self.cfg_task.env.get("use_physx_gripper_control", False):
            # ====================================================================
            # Control gripper dof with PhysX dof PD controller
            franka_handle = self.actor_handles['franka']
            for env_ptr, prop_gain, deriv_gain in zip(self.env_ptrs,
                                                      self.cfg_ctrl['gripper_prop_gains'],
                                                      self.cfg_ctrl['gripper_deriv_gains']):
                franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
                franka_dof_props['driveMode'][7:9] = gymapi.DOF_MODE_POS
                franka_dof_props['stiffness'][7:9] = prop_gain.to('cpu')
                franka_dof_props['damping'][7:9] = deriv_gain.to('cpu')
                self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

    def _set_dof_torque(self):
        """Set Franka DOF torque to move fingertips towards target pose."""

        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device)

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_torque),
                                                        gymtorch.unwrap_tensor(self.actor_ids_sim_tensors['franka']),
                                                        len(self.actor_ids_sim_tensors['franka']))

        if self.cfg_task.env.get("use_physx_gripper_control", False):
            # set target for gripper joints to use GYM's PD controller
            self.ctrl_target_dof_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
            # print(f'self.ctrl_target_gripper_dof_pos: {self.ctrl_target_gripper_dof_pos}')
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
                                                            gymtorch.unwrap_tensor(self.actor_ids_sim_tensors['franka']),
                                                            len(self.actor_ids_sim_tensors['franka']))

    def enable_gravity(self, gravity_vec):
        """Enable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity = gymapi.Vec3(*gravity_vec)
        self.gym.set_sim_params(self.sim, sim_params)

    def disable_gravity(self):
        """Disable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity = gymapi.Vec3(0, 0, 0)
        self.gym.set_sim_params(self.sim, sim_params)

    def extract_poses(self):
        """Extract poses of all bodies."""

        if not hasattr(self, 'export_pos'):
            self.export_pos = []
            self.export_rot = []
            self.frame_count = 0

        pos = self.body_pos
        rot = self.body_quat

        self.export_pos.append(pos.cpu().numpy().copy())
        self.export_rot.append(rot.cpu().numpy().copy())
        self.frame_count += 1

        if len(self.export_pos) == self.max_episode_length:
            output_dir = self.__class__.__name__
            save_dir = os.path.join('usd', output_dir)
            os.makedirs(save_dir, exist_ok=True)

            print(f'Exporting poses to {output_dir}...')
            np.save(os.path.join(save_dir, 'body_position.npy'), np.array(self.export_pos))
            np.save(os.path.join(save_dir, 'body_rotation.npy'), np.array(self.export_rot))
            print('Export completed.')
            sys.exit()

    def freeze_sim_and_render(self):
        print('Freezing simulation and just rendering. Press Ctrl+C to continue')
        while True:
            try:
                self.render()
            except:
                break
