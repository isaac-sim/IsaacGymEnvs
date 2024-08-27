'''
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Tactile Sensing Environment
----------------
A simple environment with a tactile-enabled finger that interacts with an indenting object.
It uses the tacsl visuo-tactile sensing module to generate tactile RGB image or tactile force field.
'''

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import tf_combine, tf_inverse

from isaacgymenvs.tacsl_sensors.tacsl_sensors import TactileRGBSensor, TactileFieldSensor

import math
import itertools
import numpy as np
import os
import torch

from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

import isaacgymenvs

class TactileEnv:
    def create_simulator(self, args):

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        if args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = args.num_threads
            sim_params.physx.use_gpu = args.use_gpu
            sim_params.physx.contact_collection = gymapi.ContactCollection(1)
        else:
            raise Exception('This example can only be used with PhysX')

        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception('Failed to create sim')

        # create viewer
        if not args.headless:
            # create viewer using the default camera properties
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise ValueError('*** Failed to create viewer')

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_assets(self, table_dims, indenter_name):

        self.asset_root = '../assets'
        assets = {}
        asset_file_paths = {}

        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims[0], table_dims[1], table_dims[2], asset_options)
        assets['table'] = table_asset

        # load gelsight finger asset
        if self.sensor_type == 'gelsight_r15':
            self.gelsight_asset_file = 'urdf/gelsight_description/robots/gelsight_r15_finger_decomposed.urdf'
        elif self.sensor_type == 'gs_mini':
            self.gelsight_asset_file = 'urdf/gelsight_description/robots/gsmini_finger.urdf'
        else:
            raise NotImplementedError(f'Sensor type {self.sensor_type} not recognized')
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        gelsight_asset = self.gym.load_asset(self.sim, self.asset_root, self.gelsight_asset_file, asset_options)
        assets['gelsight_finger'] = gelsight_asset
        asset_file_paths['gelsight_finger'] = self.gelsight_asset_file

        # load indenter asset
        indenter_asset_file = f'urdf/indenters/shape_sensing/{indenter_name}.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        if not self.floating_indenter:
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        indenter_asset = self.gym.load_asset(self.sim, self.asset_root, indenter_asset_file, asset_options)
        assets['indenter'] = indenter_asset
        asset_file_paths['indenter'] = indenter_asset_file

        return assets, asset_file_paths

    def create_envs_and_actors(self, table_dims):

        # add ground plane
        self._create_ground_plane()

        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print('\nCreating %d environments' % self.num_envs)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims[2])

        gelsight_pose = gymapi.Transform()
        gelsight_pose.p.x = table_pose.p.x
        gelsight_pose.p.y = table_pose.p.y
        gelsight_pose.p.z = table_dims[2] + 0.5 * 0.045 + 0.01
        gelsight_pose.r = gymapi.Quat(*R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat())
        if self.sensor_type == 'gs_mini':
            gelsight_pose.r = gymapi.Quat(*R.from_euler('xyz', [-np.pi, 0, 0]).as_quat())

        indenter_pose = gymapi.Transform()
        indenter_pose.p.x = gelsight_pose.p.x
        indenter_pose.p.y = gelsight_pose.p.y
        indenter_pose.p.z = gelsight_pose.p.z + 0.1

        actor_count = 0

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.env_ptrs.append(env_ptr)

            # add table
            table_handle = self.gym.create_actor(env_ptr, self.assets['table'], table_pose, 'table', i, 0)
            color = (222 / 255., 184 / 255., 135 / 255.)
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0,
                                          gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))

            self.actor_handles['table'] = table_handle
            self.actor_ids_sim['table'].append(actor_count)
            actor_count += 1
            # print(f'Table table_handle: \t{table_handle}')

            # add Gelsight finger
            gelsight_handle = self.gym.create_actor(env_ptr, self.assets['gelsight_finger'], gelsight_pose, 'gelsight_finger', i, 0)
            self.actor_handles['gelsight_finger'] = gelsight_handle
            self.actor_ids_sim['gelsight'].append(actor_count)
            actor_count += 1
            # print(f'Gelsight gelsight_handle: \t{gelsight_handle}')

            # add indenter
            indenter_handle = self.gym.create_actor(env_ptr, self.assets['indenter'], indenter_pose, 'indenter', i, 2)
            self.actor_handles['indenter'] = indenter_handle
            self.actor_ids_sim['indenter'].append(actor_count)
            actor_count += 1
            # print(f'Indenter indenter_handle: \t{indenter_handle}')
            indenter_rbp = self.gym.get_actor_rigid_body_properties(env_ptr, indenter_handle)[0]
            # print(f'Indenter mass: \t{indenter_rbp.mass} kg')

        # For extracting body pos/quat, force, and Jacobian
        indenter_keys = ['indenter']
        for actor_handle_key in indenter_keys:
            rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], self.actor_handles[actor_handle_key])
            assert len(rb_names) in [1, 2, 7], 'We assume that there is 2 or 8 rigid bodies in the indenter asset: ' \
                                            'Case 1: single link, free-floating body' \
                                            'Case 2: world and link, enable 1 dof motion' \
                                            ' Case 8: world and link plus 5 additional dummy links for 6 DOF motion' \
                                            '  use the name of the link (last in list) to retrieve rb id'
            rb_id_env = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles[actor_handle_key],
                                                             rb_names[-1], gymapi.DOMAIN_ENV)
            self.rb_handles[f'{actor_handle_key}_body_id_env'] = rb_id_env
        gelsight_finger_body_id_env = self.gym.find_actor_rigid_body_index(self.env_ptrs[0],
                                                                           self.actor_handles['gelsight_finger'],
                                                                           'gelsight_finger',
                                                                           gymapi.DOMAIN_ENV)
        self.rb_handles['gelsight_finger_body_id_env'] = gelsight_finger_body_id_env
        finger_elastomer_body_id_env = self.gym.find_actor_rigid_body_index(self.env_ptrs[0],
                                                                            self.actor_handles['gelsight_finger'],
                                                                            'elastomer',
                                                                            gymapi.DOMAIN_ENV)
        self.rb_handles['finger_elastomer_body_id_env'] = finger_elastomer_body_id_env

        rb_props = self.gym.get_actor_rigid_body_properties(self.env_ptrs[0], self.actor_handles['gelsight_finger'])
        rb_id_asset = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles['gelsight_finger'],
                                                           'gelsight_finger', gymapi.DOMAIN_ACTOR)
        # rb_id_asset = 0
        link_com_pos = [rb_props[rb_id_asset].com.x, rb_props[rb_id_asset].com.y, rb_props[rb_id_asset].com.z]
        link_com_pos_tensor = torch.tensor(link_com_pos, device=self.device, dtype=torch.float)
        self.rb_com['gelsight_finger'] = link_com_pos_tensor

    def acquire_tensors(self):
        """Acquire and wrap tensors. Create views."""

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)

        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 0:3]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 10:13]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]

        if self.num_dofs:
            self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
            self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.indenter_pos = self.root_pos[:, self.actor_handles['indenter'], 0:3]
        self.indenter_quat = self.root_quat[:, self.actor_handles['indenter'], 0:4]
        self.indenter_linvel = self.root_linvel[:, self.actor_handles['indenter'], 0:3]
        self.indenter_angvel = self.root_angvel[:, self.actor_handles['indenter'], 0:3]

        self.gelsight_finger_pos = self.root_pos[:, self.actor_handles['gelsight_finger'], 0:3]
        self.gelsight_finger_quat = self.root_quat[:, self.actor_handles['gelsight_finger'], 0:4]
        self.gelsight_finger_linvel = self.root_linvel[:, self.actor_handles['gelsight_finger'], 0:3]
        self.gelsight_finger_angvel = self.root_angvel[:, self.actor_handles['gelsight_finger'], 0:3]

    def refresh_tensors(self):
        """Refresh tensors."""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def reset_indenter_dof(self, dof_pos):
        """Reset DOF states and DOF targets of Indenter."""
        self.dof_pos[:] = dof_pos
        self.dof_vel[:] = 0.0
        self.ctrl_target_dof_pos[:] = dof_pos

        multi_env_ids_int32 = self.actor_ids_sim_tensors['indenter'].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(self.ctrl_target_dof_torque, device=self.device)),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32))

    def reset_indenter_pose(self, indenter_pos, indenter_quat):
        """Reset root states of indenter."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)
        indenter_actor_id_env = self.actor_handles['indenter']

        # Set root state of indenter
        self.root_pos[:, indenter_actor_id_env] = indenter_pos
        self.root_quat[:, indenter_actor_id_env] = indenter_quat

        self.root_linvel[:, indenter_actor_id_env] = 0.0
        self.root_angvel[:, indenter_actor_id_env] = 0.0

        indenter_actor_ids_sim = self.actor_ids_sim_tensors['indenter'].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(indenter_actor_ids_sim),
                                                     len(indenter_actor_ids_sim))

    def apply_action_dof_torque(self, actions):
        # Set dof torques
        self.ctrl_target_dof_torque[:] = actions

        multi_env_ids_int32 = self.actor_ids_sim_tensors['indenter'].flatten()
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_torque),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

    def apply_action_dof_pos(self, actions):
        # Set position targets
        self.ctrl_target_dof_pos[:] = actions

        multi_env_ids_int32 = self.actor_ids_sim_tensors['indenter'].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        # TODO: try velocity targets?


    def configure_actor_dofs(self, actor_handle, pd_gains=(1.0e9, 1.0), drive_mode=gymapi.DOF_MODE_POS):
        # configure actor dofs
        for env_ptr in self.env_ptrs:
            actor_dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            actor_dof_props['driveMode'][:] = drive_mode
            actor_dof_props['stiffness'][:] = pd_gains[0]
            actor_dof_props['damping'][:] = pd_gains[1]
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, actor_dof_props)

    def render_viewer(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def setup_viewer_camera(self):
        if self.viewer is None:
            return
        num_per_row = int(math.sqrt(self.num_envs))

        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_pos = gymapi.Vec3(0.874232292175293, 0.3759653568267822, 0.7455382347106934)
        cam_target = gymapi.Vec3(-4, -3, 0)
        cam_pos = gymapi.Vec3(0.6, 0.065, 0.5)
        cam_target = gymapi.Vec3(-0, 0.065, 0)
        middle_env = self.env_ptrs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def step_physics(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def clean_up(self):
        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def draw_transforms(self, pos_list_per_env, quat_list_per_env, clear_frames=True):
        if clear_frames:
            self.gym.clear_lines(self.viewer)
        num_frames_per_env = pos_list_per_env.shape[1]
        for env_id in range(len(self.env_ptrs)):
            for frame_id in range(num_frames_per_env):
                self.draw_frame_in_env(env_id, (pos_list_per_env[env_id, frame_id],
                                                quat_list_per_env[env_id, frame_id]))

    def draw_frame_in_env(self, env_id, pose, length=0.01):
        pose = gymapi.Transform(gymapi.Vec3(*pose[0]), gymapi.Quat(*pose[1]))
        axes_geom = gymutil.AxesGeometry(scale=length, pose=None)
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env_ptrs[env_id], pose)


class TactileWorld(TactileEnv, TactileRGBSensor, TactileFieldSensor):
    def __init__(self, args):
        super().__init__()
        self.gym = None
        self.sim = None
        self.viewer = None
        self.assets = []
        self.env_ptrs = []
        self.camera_properties = None

        self.sensor_type = args.sensor_type
        self.num_envs = args.num_envs
        self.table_dims = args.table_dims
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
        self.floating_indenter = args.floating_indenter

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.create_simulator(args)

        self.assets, self.asset_file_paths = self.create_assets(self.table_dims, indenter_name=args.indenter_name)

        # instantiate workbench for each env
        self.actor_handles = {}
        self.rb_handles = {}
        self.rb_com = {}
        self.actor_ids_sim = defaultdict(list)
        self.create_envs_and_actors(self.table_dims)

        # get constants from env simulation
        self.num_actors = self.gym.get_actor_count(self.env_ptrs[0])  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(self.env_ptrs[0])  # per env
        self.num_dofs = self.gym.get_env_dof_count(self.env_ptrs[0])  # per env # NOTE: self.num_dofs gets overrided here

        # For setting targets
        self.actor_ids_sim_tensors = {key: torch.tensor(self.actor_ids_sim[key], dtype=torch.int32, device=self.device)
                                      for key in self.actor_ids_sim.keys()}

        # configure movable actor dofs
        self.configure_actor_dofs(self.actor_handles['indenter'],
                                  pd_gains=(0.0, 0.0),  # PD gains only applicable when setting pos targets for gym
                                  drive_mode=gymapi.DOF_MODE_EFFORT)

        # Initialize actions
        self.ctrl_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.ctrl_target_dof_torque = torch.zeros((self.num_envs, self.num_dofs), device=self.device)

        # Set up tactile sensors
        # Use compliant dynamics for soft interpenetration of tactile elastomer
        # if compliance_stiffness is zero, it defaults to rigid contact
        compliance_stiffness = args.compliance_stiffness    # Units: force/distance = mass / seconds / seconds
        compliant_damping = 1.0                             # Units: force / (distance / seconds) = mass / seconds

        # Tactile RGB Sensor setup
        tactile_sensor_config = {
            'tactile_camera_name': 'tactile_camera',
            'actor_name': 'gelsight_finger',
            'attach_link_name': 'elastomer_tip',
            'actor_handle': self.actor_handles['gelsight_finger'],
            'elastomer_link_name': 'elastomer',
            'compliance_stiffness': compliance_stiffness,
            'compliant_damping': compliant_damping,
            'use_acceleration_spring': args.use_acceleration_spring,
            'sensor_type': args.sensor_type,
            'elastomer_parent_urdf_path': os.path.join(os.path.abspath(self.asset_root), self.gelsight_asset_file),
            'indenter_urdf_path': os.path.join(os.path.abspath(self.asset_root), self.asset_file_paths['indenter']),
            'indenter_link_name': 'indenter',
            'indenter_actor_name': 'indenter',
            'elastomer_actor_name': 'gelsight_finger',
        }
        tactile_sensor_configs = [tactile_sensor_config]

        self.set_compliant_dynamics_for_tactile_sensors(tactile_sensor_configs)

        self.camera_spec_dict = dict()
        camera_spec_dict_tactile = self.get_tactile_rgb_camera_configs(tactile_sensor_configs)
        self.camera_spec_dict.update(camera_spec_dict_tactile)
        self.tactile_camera_spec_dict = camera_spec_dict_tactile

        self.camera_handles_list, self.camera_tensors_list = [], []
        tactile_camera_handles_list, tactile_camera_tensors_list = self.create_camera_actors(camera_spec_dict_tactile)
        self.camera_handles_list += tactile_camera_handles_list
        self.camera_tensors_list += tactile_camera_tensors_list

        # Tactile Shear Field setup
        tactile_shear_field_config = dict([
            ('name', 'tactile_force_field'),
            ('elastomer_actor_name', 'gelsight_finger'), ('elastomer_link_name', 'elastomer'),
            ('elastomer_tip_link_name', 'elastomer_tip'),
            ('elastomer_parent_urdf_path', os.path.join(os.path.abspath(self.asset_root), self.gelsight_asset_file)),
            ('indenter_urdf_path', os.path.join(os.path.abspath(self.asset_root), self.asset_file_paths['indenter'])),
            ('indenter_actor_name', 'indenter'), ('indenter_link_name', 'indenter'),
            ('actor_handle', self.actor_handles['gelsight_finger']),
            ('compliance_stiffness', compliance_stiffness), ('compliant_damping', compliant_damping),
            ('use_acceleration_spring', args.use_acceleration_spring)
        ])
        tactile_shear_field_configs = [tactile_shear_field_config]
        self.set_compliant_dynamics_for_tactile_sensors(tactile_shear_field_configs)
        self.sdf_tensor = self.setup_tactile_force_field(args.sdf_tool, args.num_tactile_rows, args.num_tactile_cols,
                                                         tactile_shear_field_configs)

        # start GPU
        self.gym.prepare_sim(self.sim)

        # setup tensors
        self.acquire_tensors()
        self.refresh_tensors()

        # Initialize tactile sensors
        self.initialize_tactile_rgb_camera()

        # Shear force initialization
        num_divs = [args.num_tactile_rows, args.num_tactile_cols]
        self.initialize_penalty_based_tactile(num_divs=num_divs)

    def get_elastomer_dimension(self):
        robot = URDF.load(os.path.join(self.asset_root, self.gelsight_asset_file))
        mesh = robot.link_map['elastomer'].visuals[0].geometry.mesh.meshes[0]
        elastomer_dims = np.diff(mesh.bounds, axis=0).squeeze()
        return elastomer_dims

    def generate_grid_on_elastomer(self, num_divs=2):

        elastomer_dims = self.get_elastomer_dimension()

        slim_axis = np.argmin(elastomer_dims)
        planar_grid_points = []
        for axis_i in range(3):
            if axis_i == slim_axis:
                planar_grid_points.append([0])
            else:
                axis_grid_points = np.linspace(-elastomer_dims[axis_i] / 2, elastomer_dims[axis_i] / 2, num_divs + 2)
                planar_grid_points.append(axis_grid_points[1:-1])  # leave out the extreme corners

        grid_corners_3D = itertools.product(planar_grid_points[0], planar_grid_points[1], planar_grid_points[2])
        grid_corners_3D = np.array(list(grid_corners_3D))

        # place the planar grid points centered on the elastomer tip point (right at the centre of the elastomer)
        num_grid_points = grid_corners_3D.shape[0]
        pos_list_per_env = torch.zeros((self.num_envs, num_grid_points, 3), dtype=torch.float32, device=self.device)
        quat_list_per_env = torch.zeros((self.num_envs, num_grid_points, 4), dtype=torch.float32, device=self.device)
        gelsight_handle = self.actor_handles['gelsight_finger']
        grid_corners_3D_tensor = torch.tensor(grid_corners_3D, dtype=torch.float32, device=self.device)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        elastomer_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], gelsight_handle, 'elastomer')
        elastomer_tip_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], gelsight_handle,
                                                                          'elastomer_tip')

        elastomer_to_tip_link_trans = tf_combine(*tf_inverse(self.body_quat[:, elastomer_link_handle],
                                                             self.body_pos[:, elastomer_link_handle]),
                                                 self.body_quat[:, elastomer_tip_link_handle],
                                                 self.body_pos[:, elastomer_tip_link_handle])
        for idx, grid_corner_offset in enumerate(grid_corners_3D_tensor):
            quat_list_per_env[:, idx], pos_list_per_env[:, idx] = tf_combine(self.body_quat[:, elastomer_link_handle],
                                                                             self.body_pos[:, elastomer_link_handle],
                                                                             identity_quat,
                                                                             grid_corner_offset.repeat(self.num_envs,
                                                                                                       1))
            quat_list_per_env[:, idx], pos_list_per_env[:, idx] = tf_combine(quat_list_per_env[:, idx],
                                                                             pos_list_per_env[:, idx],
                                                                             elastomer_to_tip_link_trans[0],
                                                                             elastomer_to_tip_link_trans[1])

        return pos_list_per_env, quat_list_per_env

    def compute_indenter_start_poses_for_elastomer_nodes(self, elastomer_node_pos_list_per_env,
                                                         elastomer_node_quat_list_per_env,
                                                         distance_to_sensor=0.01, rotation=(0, 0, -np.pi)):
        num_grid_points = elastomer_node_pos_list_per_env.shape[1]
        pos_list_per_env = torch.zeros((self.num_envs, num_grid_points, 3), dtype=torch.float32, device=self.device)
        quat_list_per_env = torch.zeros((self.num_envs, num_grid_points, 4), dtype=torch.float32, device=self.device)

        clearance_offset = torch.tensor([0, 0, distance_to_sensor],
                                        device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        indenter_dof_reorient = torch.tensor(R.from_euler('xyz', rotation).as_quat(),
                                             device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        for idx in range(num_grid_points):
            quat_list_per_env[:, idx], pos_list_per_env[:, idx] = tf_combine(elastomer_node_quat_list_per_env[:, idx],
                                                                             elastomer_node_pos_list_per_env[:, idx],
                                                                             indenter_dof_reorient,
                                                                             clearance_offset)
        return pos_list_per_env, quat_list_per_env

    def get_link_pose_per_env(self, actor_handle, link_name):
        link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], actor_handle, link_name)
        return self.body_pos[:, link_handle], self.body_quat[:, link_handle]

    def get_actor_names_for_all_env_rigid_bodies(self, env_id):
        env_handle = self.env_ptrs[env_id]
        num_actors = self.gym.get_actor_count(env_handle)
        rb_id_env_to_actor_id = dict()
        rb_id_env_to_actor_name = dict()
        rb_id_env_to_actor_and_rb_name = dict()
        for actor_id in range(num_actors):
            actor_handle = self.gym.get_actor_handle(env_handle, actor_id)
            num_rigid_bodies = self.gym.get_actor_rigid_body_count(env_handle, actor_handle)
            rb_names_in_actor = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
            for rb in range(num_rigid_bodies):
                rb_id_env = self.gym.get_actor_rigid_body_index(env_handle, actor_handle, rb, gymapi.DOMAIN_ENV)
                rb_id_env_to_actor_id[rb_id_env] = actor_id
                rb_id_env_to_actor_name[rb_id_env] = self.gym.get_actor_name(env_handle, actor_handle)
                actor_name = self.gym.get_actor_name(env_handle, actor_handle)
                rb_name = rb_names_in_actor[rb]
                rb_id_env_to_actor_and_rb_name[rb_id_env] = (actor_name, rb_name)
        return rb_id_env_to_actor_id, rb_id_env_to_actor_name, rb_id_env_to_actor_and_rb_name

    def freeze_sim_and_render(self):
        while True:
            try:
                self.render_viewer()
            except:
                break

    def set_up_keyboard(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "x_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "x_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "y_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "y_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "z_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "z_down")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "roll_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "roll_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "pitch_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "pitch_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "yaw_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "yaw_down")


def parse_args():

    # Add custom arguments
    custom_parameters = [
        {'name': '--num_envs', 'type': int, 'default': 256, 'help': 'Number of environments to create'},
        {'name': '--table_dims', 'type': int, 'default': [0.6, 1.0, 0.4], 'help': 'Dimension of table'},
        {'name': '--compliance_stiffness', 'type': float, 'default': 200.0, 'help': 'Compliant contact stiffness'},
        {'name': '--use_acceleration_spring', 'action': 'store_true', 'help': 'When True, use acceleration_spring for compliant contacts.'},
        {'name': '--sensor_type', 'type': str, 'default': 'gelsight_r15', 'help': 'Name of indenter. Choose from {gelsight_r15, gs_mini}'},
        {'name': '--indenter_name', 'type': str, 'default': 'factory_nut_m8_loose_subdiv_3x_6DOF', 'help': 'Name of indenter'},
        {'name': '--distance_to_sensor', 'type': float, 'default': 0.03, 'help': 'Initial indenter distance to gelsight sensor'},
        {'name': '--sdf_tool', 'type': str, 'default': 'physx', 'choices': ['trimesh', 'pysdf', 'physx'], 'help': 'which package to compute sdf'},
        {'name': '--num_tactile_rows', 'type': int, 'default': 20, 'help': '#rows of tactile marker points'},
        {'name': '--num_tactile_cols', 'type': int, 'default': 25, 'help': '#cols of tactile marker points'},
        {'name': '--render', 'action': 'store_true', 'help': "whether to render the simulation or not"},
        {'name': '--record', 'action': 'store_true', 'help': "whether to record the tactile images"},
        {'name': '--use_tactile_rgb', 'action': 'store_true', 'help': "whether to use tactile RGB sensor"},
        {'name': '--use_tactile_ff', 'action': 'store_true', 'help': "whether to use tactile force field sensor"},
        {'name': '--floating_indenter', 'action': 'store_true', 'help': "flag to make indenter a free floating body"},
        {'name': '--action_duration', 'type': float, 'default': 1.0, 'help': 'Number of seconds for each indenting action'},
    ]
    args = gymutil.parse_arguments(
        headless=True,
        description='Shape Sensing Example showing visuo-tactile sensing and shear force field.',
        custom_parameters=custom_parameters,
    )
    return args

