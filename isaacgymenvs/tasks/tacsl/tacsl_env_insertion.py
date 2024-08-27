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

"""TacSL: class for insertion env.

Inherits TacSL base class and abstract environment class. Inherited by insertion task class. Not directly executed.

Configuration defined in TacSLEnvInsertion.yaml. Asset info defined in industreal_asset_info_pegs.yaml.
"""

from collections import defaultdict
import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgym.torch_utils import tf_combine
from isaacgymenvs.tacsl_sensors.tacsl_sensors import CameraSensor, TactileRGBSensor, TactileFieldSensor
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from isaacgymenvs.tasks.tacsl.tacsl_base import TacSLBase


class TacSLSensors(TactileFieldSensor, TactileRGBSensor, CameraSensor):

    def get_regular_camera_specs(self):
        # TODO: Maybe this should go inside the task files, as it depends on task configs e.g. self.cfg_task.env.use_camera

        camera_spec_dict = {}
        if self.cfg_task.env.use_camera:
            camera_spec_dict = {c_cfg.name: c_cfg for c_cfg in self.cfg_task.env.camera_configs}
        return camera_spec_dict

    def _compose_tactile_image_configs(self):
        tactile_sensor_config = {
            'tactile_camera_name': 'left_tactile_camera',
            'actor_name': 'franka',
            'actor_handle': self.actor_handles['franka'],
            'attach_link_name': 'elastomer_tip_left',
            'elastomer_link_name': 'elastomer_left',
            'compliance_stiffness': self.cfg_task.env.compliance_stiffness,
            'compliant_damping': self.cfg_task.env.compliant_damping,
            'use_acceleration_spring': False,
            'sensor_type': 'gelsight_r15'
        }
        tactile_sensor_config_left = tactile_sensor_config.copy()
        tactile_sensor_config_right = tactile_sensor_config.copy()
        tactile_sensor_config_right['tactile_camera_name'] = 'right_tactile_camera'
        tactile_sensor_config_right['attach_link_name'] = 'elastomer_tip_right'
        tactile_sensor_config_right['elastomer_link_name'] = 'elastomer_right'
        tactile_sensor_configs = [tactile_sensor_config_left, tactile_sensor_config_right]
        return tactile_sensor_configs

    def _compose_tactile_force_field_configs(self):
        plug_rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], self.plug_actor_id_env)
        tactile_shear_field_config = dict([
            ('name', 'tactile_force_field_left'),
            ('elastomer_actor_name', 'franka'), ('elastomer_link_name', 'elastomer_left'),
            ('elastomer_tip_link_name', 'elastomer_tip_left'),
            ('elastomer_parent_urdf_path', self.asset_file_paths["franka"]),
            ('indenter_urdf_path', self.asset_file_paths["plug"]),
            ('indenter_actor_name', 'plug'), ('indenter_link_name', plug_rb_names[0]),
            ('actor_handle', self.actor_handles['franka']),
            ('compliance_stiffness', self.cfg_task.env.compliance_stiffness),
            ('compliant_damping', self.cfg_task.env.compliant_damping),
            ('use_acceleration_spring', False)
        ])
        tactile_shear_field_config_left = tactile_shear_field_config.copy()
        tactile_shear_field_config_right = tactile_shear_field_config.copy()
        tactile_shear_field_config_right['name'] = 'tactile_force_field_right'
        tactile_shear_field_config_right['elastomer_link_name'] = 'elastomer_right'
        tactile_shear_field_config_right['elastomer_tip_link_name'] = 'elastomer_tip_right'
        tactile_shear_field_configs = [tactile_shear_field_config_left, tactile_shear_field_config_right]
        return tactile_shear_field_configs

    def get_tactile_force_field_tensors_dict(self):

        tactile_force_field_dict_raw = self.get_tactile_shear_force_fields()
        tactile_force_field_dict_processed = dict()
        nrows, ncols = self.cfg_task.env.num_shear_rows, self.cfg_task.env.num_shear_cols

        debug = False   # Debug visualization
        for k in tactile_force_field_dict_raw:
            penetration_depth, tactile_normal_force, tactile_shear_force = tactile_force_field_dict_raw[k]
            tactile_force_field = torch.cat(
                (tactile_normal_force.view((self.num_envs, nrows, ncols, 1)),
                 tactile_shear_force.view((self.num_envs, nrows, ncols, 2))),
                dim=-1)
            tactile_force_field_dict_processed[k] = tactile_force_field

            if debug:
                env_viz_id = 0
                tactile_image = visualize_tactile_shear_image(
                    tactile_normal_force[env_viz_id].view((nrows, ncols)).cpu().numpy(),
                    tactile_shear_force[env_viz_id].view((nrows, ncols, 2)).cpu().numpy(),
                    normal_force_threshold=0.0008,
                    shear_force_threshold=0.0008)
                cv2.imshow(f'Force Field {k}', tactile_image.swapaxes(0, 1))

                penetration_depth_viz = visualize_penetration_depth(
                    penetration_depth[env_viz_id].view((nrows, ncols)).cpu().numpy(),
                    resolution=5, depth_multiplier=300.)
                cv2.imshow(f'FF Penetration Depth {k}', penetration_depth_viz.swapaxes(0, 1))
        return tactile_force_field_dict_processed

    def _create_sensors(self):
        self.camera_spec_dict = dict()
        self.camera_handles_list = []
        self.camera_tensors_list = []
        if self.cfg_task.env.use_isaac_gym_tactile:
            tactile_sensor_configs = self._compose_tactile_image_configs()
            self.set_compliant_dynamics_for_tactile_sensors(tactile_sensor_configs)
            camera_spec_dict_tactile = self.get_tactile_rgb_camera_configs(tactile_sensor_configs)
            self.camera_spec_dict.update(camera_spec_dict_tactile)

        if self.cfg_task.env.use_camera:
            camera_spec_dict = self.get_regular_camera_specs()
            self.camera_spec_dict.update(camera_spec_dict)

        if self.camera_spec_dict:
            # tactile cameras created along with other cameras in create_camera_actors
            camera_handles_list, camera_tensors_list = self.create_camera_actors(self.camera_spec_dict)
            self.camera_handles_list += camera_handles_list
            self.camera_tensors_list += camera_tensors_list

        if self.cfg_task.env.get('use_shear_force', False):
            tactile_ff_configs = self._compose_tactile_force_field_configs()
            self.set_compliant_dynamics_for_tactile_sensors(tactile_ff_configs)
            self.sdf_tool = 'physx'
            self.sdf_tensor = self.setup_tactile_force_field(self.sdf_tool,
                                                             self.cfg_task.env.num_shear_rows,
                                                             self.cfg_task.env.num_shear_cols,
                                                             tactile_ff_configs)


class TacSLEnvInsertion(TacSLBase, TacSLSensors, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
        self.nominal_tactile = None

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/TacSLEnvInsertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        subassemblies_override_str = f'task.env.desired_subassemblies=[{",".join(self.cfg_task.env.desired_subassemblies)}]'
        self.cfg_env = hydra.compose(config_name=config_path,
                                     overrides=[subassemblies_override_str])
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = f'../../assets/tacsl/yaml/{self.cfg_task.env.asset_info_filename}'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['tacsl']['yaml']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.asset_info_franka_table.table_depth * 0.6,
                            -self.asset_info_franka_table.table_width * 0.6,
                            0.0)
        upper = gymapi.Vec3(self.asset_info_franka_table.table_depth * 0.6,
                            self.asset_info_franka_table.table_width * 0.6,
                            self.asset_info_franka_table.table_height)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        self.assets = dict()
        self.asset_file_paths = dict()
        self.assets['franka'], self.assets['table'] = self.import_franka_assets()
        self._import_env_assets()
        self._create_actors(lower, upper, num_per_row)
        self.parse_controller_spec()

        self._create_sensors()

    def _import_env_assets(self):
        """Set plug and socket asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets')

        plug_options = gymapi.AssetOptions()
        plug_options.flip_visual_attachments = False
        plug_options.fix_base_link = False
        plug_options.thickness = 0.0  # default = 0.02
        plug_options.armature = 0.0  # default = 0.0
        plug_options.use_physx_armature = True
        plug_options.linear_damping = 0.0  # default = 0.0
        plug_options.max_linear_velocity = 1000.0  # default = 1000.0
        plug_options.angular_damping = 0.0  # default = 0.5
        plug_options.max_angular_velocity = 64.0  # default = 64.0
        plug_options.disable_gravity = False
        plug_options.enable_gyroscopic_forces = True
        plug_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        plug_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            plug_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        plug_options_tactile = gymapi.AssetOptions()
        plug_options_tactile.flip_visual_attachments = False
        plug_options_tactile.fix_base_link = True

        socket_options = gymapi.AssetOptions()
        socket_options.flip_visual_attachments = False
        socket_options.fix_base_link = True
        socket_options.thickness = 0.0  # default = 0.02
        socket_options.armature = 0.0  # default = 0.0
        socket_options.use_physx_armature = True
        socket_options.linear_damping = 0.0  # default = 0.0
        socket_options.max_linear_velocity = 1000.0  # default = 1000.0
        socket_options.angular_damping = 0.0  # default = 0.5
        socket_options.max_angular_velocity = 64.0  # default = 64.0
        socket_options.disable_gravity = False
        socket_options.enable_gyroscopic_forces = True
        socket_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        socket_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            socket_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        plug_assets = []
        socket_assets = []

        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_insertion[subassembly])
            plug_file_parent_dir = os.path.join(self.asset_info_insertion[subassembly][components[0]]['urdf_par_dir'], 'urdf')
            plug_file = self.asset_info_insertion[subassembly][components[0]]['urdf_path'] + '.urdf'
            socket_file_parent_dir = os.path.join(self.asset_info_insertion[subassembly][components[1]]['urdf_par_dir'], 'urdf')
            socket_file = self.asset_info_insertion[subassembly][components[1]]['urdf_path'] + '.urdf'
            plug_options.density = self.asset_info_insertion[subassembly][components[0]]['density']
            socket_options.density = self.asset_info_insertion[subassembly][components[1]]['density']
            plug_asset_file_path = os.path.join(os.path.abspath(urdf_root), plug_file_parent_dir, plug_file)
            self.asset_file_paths['plug'] = plug_asset_file_path
            plug_asset = self.gym.load_asset(self.sim, os.path.join(urdf_root, plug_file_parent_dir), plug_file, plug_options)
            socket_asset = self.gym.load_asset(self.sim, os.path.join(urdf_root, socket_file_parent_dir), socket_file, socket_options)
            plug_assets.append(plug_asset)
            socket_assets.append(socket_asset)

        finger_options = gymapi.AssetOptions()
        finger_options.flip_visual_attachments = True
        finger_options.fix_base_link = True

        self.assets['plugs'] = plug_assets
        self.assets['sockets'] = socket_assets

    def _create_actors(self, lower, upper, num_per_row):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = 0.0
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = self.asset_info_franka_table.robot_base_to_table_offset_x
        table_pose.p.y = 0.0
        table_pose.p.z = -self.asset_info_franka_table.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.table_handles = []
        self.shape_ids = []
        self.table_actor_ids_sim = []  # within-sim indices
        self.env_subassembly_id = []
        self.actor_handles = {}
        self.actor_ids_sim = defaultdict(list)
        self.rbs_com = defaultdict(list)
        actor_count = 0

        self.plug_lengths = []
        self.socket_heights = []
        self.socket_diameters = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, self.assets['franka'], franka_pose, 'franka',
                                                      i + self.num_envs, 0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, self.assets['franka'], franka_pose, 'franka', i, 0, 0)
            self.actor_handles['franka'] = franka_handle
            self.actor_ids_sim['franka'].append(actor_count)
            actor_count += 1

            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_insertion[subassembly])
            self.env_subassembly_id.append(j)

            plug_pose = gymapi.Transform()
            plug_pose.p.x = self.cfg_task.randomize.socket_pos_xyz_initial[0]
            plug_pose.p.y = self.cfg_task.randomize.socket_pos_xyz_initial[1]
            plug_pose.p.z = self.cfg_task.randomize.socket_pos_xyz_initial[2]
            plug_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            plug_handle = self.gym.create_actor(env_ptr, self.assets['plugs'][j], plug_pose, 'plug', i, 0, 0)
            self.actor_handles['plug'] = plug_handle
            self.actor_ids_sim['plug'].append(actor_count)
            actor_count += 1

            plug_length = self.asset_info_insertion[subassembly][components[0]]['length']
            self.plug_lengths.append(plug_length)

            socket_pose = gymapi.Transform()
            socket_pose.p.x = self.cfg_task.randomize.socket_pos_xyz_initial[0]
            socket_pose.p.y = self.cfg_task.randomize.socket_pos_xyz_initial[1]
            socket_pose.p.z = self.cfg_task.randomize.socket_pos_xyz_initial[2]
            socket_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            socket_handle = self.gym.create_actor(env_ptr, self.assets['sockets'][j], socket_pose, 'socket', i, 0, 0)
            self.actor_handles['socket'] = socket_handle
            self.actor_ids_sim['socket'].append(actor_count)
            actor_count += 1

            socket_height = self.asset_info_insertion[subassembly][components[1]]['height']
            self.socket_heights.append(socket_height)
            if 'diameter' in self.asset_info_insertion[subassembly][components[1]]:
                socket_diameter = self.asset_info_insertion[subassembly][components[1]]['diameter']
            elif 'width' in self.asset_info_insertion[subassembly][components[1]]:
                socket_diameter = max(self.asset_info_insertion[subassembly][components[1]]['width'],
                                      self.asset_info_insertion[subassembly][components[1]]['depth'])
            else:
                raise NotImplementedError
            self.socket_diameters.append(socket_diameter)

            table_handle = self.gym.create_actor(env_ptr, self.assets['table'], table_pose, 'table', i, 0, 0)
            self.actor_handles['table'] = table_handle
            self.actor_ids_sim['table'].append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            rb_ids = [link7_id, hand_id, left_finger_id, right_finger_id]
            rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(self.assets['franka'])
            self.shape_ids = [rb_shape_indices[rb_id].start for rb_id in rb_ids]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            plug_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, plug_handle)
            plug_shape_props[0].friction = self.asset_info_insertion[subassembly][components[0]]['friction']
            plug_shape_props[0].rolling_friction = 0.0  # default = 0.0
            plug_shape_props[0].torsion_friction = 0.0  # default = 0.0
            plug_shape_props[0].restitution = 0.0  # default = 0.0
            plug_shape_props[0].compliance = 0.0  # default = 0.0
            plug_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, plug_handle, plug_shape_props)

            socket_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, socket_handle)
            socket_shape_props[0].friction = self.asset_info_insertion[subassembly][components[1]]['friction']
            socket_shape_props[0].rolling_friction = 0.0  # default = 0.0
            socket_shape_props[0].torsion_friction = 0.0  # default = 0.0
            socket_shape_props[0].restitution = 0.0  # default = 0.0
            socket_shape_props[0].compliance = 0.0  # default = 0.0
            socket_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, socket_handle, socket_shape_props)

            for asset_name, asset_handle in zip(['plug', 'socket'], [plug_handle, socket_handle]):
                rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, asset_handle)
                rb_id_asset = 0
                link_com_pos = [rb_props[rb_id_asset].com.x, rb_props[rb_id_asset].com.y, rb_props[rb_id_asset].com.z]
                self.rbs_com[asset_name].append(link_com_pos)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)

        if self.cfg_task.env.use_compliant_contact:
            # Set compliance params
            self.set_elastomer_compliance(self.cfg_task.env.compliance_stiffness, self.cfg_task.env.compliant_damping)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.actor_ids_sim_tensors = {key: torch.tensor(self.actor_ids_sim[key], dtype=torch.int32, device=self.device)
                                      for key in self.actor_ids_sim.keys()}

        # For extracting root pos/quat
        self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
        self.plug_actor_id_env = self.gym.find_actor_index(env_ptr, 'plug', gymapi.DOMAIN_ENV)
        self.socket_actor_id_env = self.gym.find_actor_index(env_ptr, 'socket', gymapi.DOMAIN_ENV)
        self.rbs_com_tensors = {key: torch.tensor(self.rbs_com[key], dtype=torch.float, device=self.device) for key in
                                self.rbs_com.keys()}

        # For extracting body pos/quat, force, and Jacobian
        plug_rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], self.plug_actor_id_env)
        assert len(plug_rb_names) == 1, 'We assume that there is a single rigid body in the plug asset and use the name to retrieve rb id'
        self.plug_body_id_env = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.plug_actor_id_env,
                                                                     plug_rb_names[0], gymapi.DOMAIN_ENV)
        socket_rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], self.socket_actor_id_env)
        assert len(socket_rb_names) == 1, 'We assume that there is a single rigid body in the socket asset and use the name to retrieve rb id'
        self.socket_body_id_env = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.socket_actor_id_env,
                                                                       socket_rb_names[0], gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.left_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                               'panda_leftfingertip',
                                                                               gymapi.DOMAIN_ENV)
        self.right_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                'panda_rightfingertip',
                                                                                gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)
        self.hand_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                           gymapi.DOMAIN_ACTOR)
        self.left_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                  'panda_leftfinger',
                                                                                  gymapi.DOMAIN_ACTOR)
        self.right_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_rightfinger',
                                                                                   gymapi.DOMAIN_ACTOR)
        self.left_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                     'panda_leftfingertip',
                                                                                     gymapi.DOMAIN_ACTOR)
        self.right_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                      'panda_rightfingertip',
                                                                                      gymapi.DOMAIN_ACTOR)
        self.fingertip_centered_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                         'panda_fingertip_centered',
                                                                                         gymapi.DOMAIN_ACTOR)
        self.table_body_id = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles['table'],
                                                                  'box', gymapi.DOMAIN_ENV)
        self.franka_body_names = self.gym.get_actor_rigid_body_names(env_ptr, franka_handle)
        self.franka_body_ids_env = dict()
        for b_name in self.franka_body_names:
            self.franka_body_ids_env[b_name] = self.gym.find_actor_rigid_body_index(self.env_ptrs[0],
                                                                                    self.actor_handles['franka'],
                                                                                    b_name, gymapi.DOMAIN_ENV)

        # For setting initial state
        self.plug_lengths = torch.tensor(self.plug_lengths, device=self.device).unsqueeze(-1)
        self.socket_heights = torch.tensor(self.socket_heights, device=self.device).unsqueeze(-1)
        self.socket_diameters = torch.tensor(self.socket_diameters, device=self.device).unsqueeze(-1)

    def set_elastomer_compliance(self, compliance_stiffness, compliant_damping):
        for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
            self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                              elastomer_link_name=elastomer_link_name,
                                              compliance_stiffness=compliance_stiffness,
                                              compliant_damping=compliant_damping,
                                              use_acceleration_spring=False)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.franka_base_pos = self.root_pos[:, self.franka_actor_id_env, 0:3]
        self.franka_base_quat = self.root_quat[:, self.franka_actor_id_env, 0:4]

        self.plug_pos = self.root_pos[:, self.plug_actor_id_env, 0:3]
        self.plug_quat = self.root_quat[:, self.plug_actor_id_env, 0:4]
        self.plug_linvel = self.root_linvel[:, self.plug_actor_id_env, 0:3]
        self.plug_angvel = self.root_angvel[:, self.plug_actor_id_env, 0:3]

        self.socket_pos = self.root_pos[:, self.socket_actor_id_env, 0:3]
        self.socket_quat = self.root_quat[:, self.socket_actor_id_env, 0:4]

        self.plug_com_pos = torch.zeros_like(self.plug_pos)
        self.plug_com_quat = self.plug_quat  # always equal
        self.plug_com_linvel = self.plug_linvel   # By default, isaac gymm returns velocity at centre of mass
        self.plug_com_angvel = self.plug_angvel  # always equal
        self.plug_origin_linvel = torch.zeros_like(self.plug_linvel)

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)
        _, self.plug_com_pos = tf_combine(self.plug_quat, self.plug_pos, identity_quat, self.rbs_com_tensors['plug'])

        self.plug_com_linvel = self.plug_linvel     # By default, isaac gymm returns velocity at centre of mass
        self.plug_origin_linvel = self.plug_linvel + torch.cross(self.plug_angvel, -self.rbs_com_tensors['plug'], dim=1)
