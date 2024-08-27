'''
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Ball Rolling tactile-sensing world
----------------
A simple tactile sensing world with a box-shaped tactile sensor interacting with a ball.
'''
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import tf_combine
from isaacgym import torch_utils as tu

import math
import numpy as np
import os
import torch
from urdfpy import URDF

from scipy.spatial.transform import Rotation as R
from collections import defaultdict

class BallRollingTactileWorld:
    def __init__(self, args):
        self.gym = None
        self.sim = None
        self.viewer = None
        self.assets = []
        self.env_ptrs = []
        self.camera_handles = []
        self.camera_properties = None

        self.num_envs = args.num_envs
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'

        # Acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.create_simulator(args)

        self.assets = self.create_assets()

        # Instantiate workbench for each environment
        self.actor_handles = {}
        self.rb_handles = {}
        self.rb_com = {}
        self.actor_ids_sim = defaultdict(list)
        self.create_envs_and_actors()

        # Optionally use compliant dynamics for soft interpenetration of tactile elastomer
        self.use_compliant_dyn = args.use_compliant_dyn
        if self.use_compliant_dyn:
            compliance_stiffness = 5.942314/0.00058/200/1.0    # [sphere] if max force = xx, and max deflection is yy, stiffness = xx/yy
            compliant_damping = 0.1 * compliance_stiffness       # Units: force / (distance / seconds) = mass / seconds
            self.configure_compliant_dynamics(compliance_stiffness, compliant_damping)
        
        # Get constants from environment simulation
        self.num_actors = self.gym.get_actor_count(self.env_ptrs[0])  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(self.env_ptrs[0])  # per env
        self.num_dofs = self.gym.get_env_dof_count(self.env_ptrs[0])  # per env # NOTE: self.num_dofs gets overrided here

        # For setting targets
        self.actor_ids_sim_tensors = {key: torch.tensor(self.actor_ids_sim[key], dtype=torch.int32, device=self.device)
                                      for key in self.actor_ids_sim.keys()}

        # Set up tactile SDF tensors
        if args.sdf_tool == 'physx':
            num_queries_per_env = args.num_tactile_rows * args.num_tactile_cols
            _sdf_tensor = self.gym.acquire_sdf_view_tensor(self.sim, 1, num_queries_per_env)
            self.sdf_tensor = gymtorch.wrap_tensor(_sdf_tensor)
        
        # Start GPU
        self.gym.prepare_sim(self.sim)

        # Setup tensors
        self.acquire_tensors()
        self.refresh_tensors()

        # Initialize actions
        self.ctrl_target_dof_torque = torch.zeros((self.num_envs, self.num_dofs), device=self.device)

    def create_simulator(self, args):

        # Configure sim
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

        # Create simulation
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception('Failed to create sim')

        # Create viewer
        if not args.headless:
            # create viewer using the default camera properties
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise ValueError('*** Failed to create viewer')

    def create_assets(self):

        self.asset_root = '../assets'
        assets = {}

        # Load pad asset
        self.pad_asset_file = 'urdf/ball_rolling/pad.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.density = 1000
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        pad_asset = self.gym.load_asset(self.sim, self.asset_root, self.pad_asset_file, asset_options)

        assets['pad'] = pad_asset

        # Load ball asset
        self.ball_asset_file = f'urdf/ball_rolling/ball.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.density = 1000.0
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        ball_asset = self.gym.load_asset(self.sim, self.asset_root, self.ball_asset_file, asset_options)
        assets['ball'] = ball_asset

        return assets

    def create_envs_and_actors(self):

        # add ground plane
        self._create_ground_plane()

        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 0.1
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print('Creating %d environments' % self.num_envs)

        pad_pose = gymapi.Transform()
        pad_pose.p.x = 0.
        pad_pose.p.y = 0.
        pad_pose.p.z = 0.06

        ball_pose = gymapi.Transform()
        ball_pose.p.x = 0.
        ball_pose.p.y = 0.
        ball_pose.p.z = 0.02

        actor_count = 0

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.env_ptrs.append(env_ptr)

            # Add pad
            pad_handle = self.gym.create_actor(env_ptr, self.assets['pad'], pad_pose, 'pad', i, 0)
            self.actor_handles['pad'] = pad_handle
            self.actor_ids_sim['pad'].append(actor_count)
            actor_count += 1

            # Add ball
            ball_handle = self.gym.create_actor(env_ptr, self.assets['ball'], ball_pose, 'ball', i, 2)
            self.actor_handles['ball'] = ball_handle
            self.actor_ids_sim['ball'].append(actor_count)
            actor_count += 1

        # For extracting body pos/quat, force, and Jacobian
        for actor_handle_key in ['pad', 'ball']:
            rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], self.actor_handles[actor_handle_key])
            rb_id_env = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles[actor_handle_key],
                                                             rb_names[-1], gymapi.DOMAIN_ENV)
            self.rb_handles[f'{actor_handle_key}_body_id_env'] = rb_id_env
        
        # configure movable actor dofs, force control
        self.configure_actor_dofs(self.actor_handles['pad'],
                                    pd_gains=(0.0, 0.0),  # PD gains only applicable when setting pos targets for gym
                                    drive_mode=gymapi.DOF_MODE_EFFORT)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def configure_compliant_dynamics(self, compliance_stiffness=0.0, compliant_damping=0.0):
        actor_handle = self.actor_handles['pad']

        for env_ptr in self.env_ptrs:
            rs_props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)

            rs_props[0].compliance = compliance_stiffness
            rs_props[0].compliant_damping = compliant_damping
            self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, rs_props)

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

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.pad_pos = self.root_pos[:, self.actor_handles['pad'], 0:3]
        self.pad_quat = self.root_quat[:, self.actor_handles['pad'], 0:4]
        self.pad_linvel = self.root_linvel[:, self.actor_handles['pad'], 0:3]
        self.pad_angvel = self.root_angvel[:, self.actor_handles['pad'], 0:3]

        self.ball_finger_pos = self.root_pos[:, self.actor_handles['ball'], 0:3]
        self.ball_finger_quat = self.root_quat[:, self.actor_handles['ball'], 0:4]
        self.ball_finger_linvel = self.root_linvel[:, self.actor_handles['ball'], 0:3]
        self.ball_finger_angvel = self.root_angvel[:, self.actor_handles['ball'], 0:3]

    def refresh_tensors(self):
        """Refresh tensors."""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
    
    def configure_actor_dofs(self, actor_handle, pd_gains=(1.0e9, 1.0), drive_mode=gymapi.DOF_MODE_POS):
        # configure actor dofs
        for env_ptr in self.env_ptrs:
            actor_dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            actor_dof_props['driveMode'][:] = drive_mode
            actor_dof_props['stiffness'][:] = pd_gains[0]
            actor_dof_props['damping'][:] = pd_gains[1]
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, actor_dof_props)
    
    def apply_action_dof_torque(self, actions):
        # Set DOF torques
        self.ctrl_target_dof_torque[:] = actions

        multi_env_ids_int32 = self.actor_ids_sim_tensors['pad'].flatten()
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_torque),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
    
    def step_physics(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def setup_viewer_camera(self):
        if self.viewer is None:
            return
        num_per_row = int(math.sqrt(self.num_envs))

        # point camera at middle env
        cam_pos = gymapi.Vec3(0.874232292175293, 0.3759653568267822, 0.7455382347106934)
        cam_target = gymapi.Vec3(0, 0, 0.02)
        middle_env = self.env_ptrs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

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

    def render_viewer(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def freeze_sim_and_render(self):
        while True:
            try:
                self.render_viewer()
            except:
                break
    
    def get_transform(self, link_handle_name):
        link_handle = self.rb_handles[link_handle_name]
        pos = self.body_pos[:, link_handle]
        quat = self.body_quat[:, link_handle]
        return quat, pos
    
    def get_velocity(self, link_handle_name):
        link_handle = self.rb_handles[link_handle_name]
        linvel = self.body_linvel[:, link_handle]
        angvel = self.body_angvel[:, link_handle]
        return angvel, linvel

    def generate_tactile_points(self, num_divs=[20, 20]):
        pad_dims = [0.05, 0.05, 0.01]

        dx = pad_dims[0] / num_divs[0]
        dy = pad_dims[1] / num_divs[1]
        tactile_points = []
        for i in range(num_divs[0]):
            for j in range(num_divs[1]):
                tactile_points.append([dx * i + dx / 2. - pad_dims[0] / 2., dy * j + dy / 2. - pad_dims[1] / 2., -pad_dims[2] / 2.])
        tactile_points_pos_tensor = torch.tensor(tactile_points, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # rotation = (np.pi, 0, 0) # NOTE [Jie]: assume tactile frame rotation are all the same
        rotation = (0, 0, 0) # NOTE [Jie]: assume tactile frame rotation are all the same
        tactile_points_quat = R.from_euler('xyz', rotation).as_quat()
        tactile_points_quat_tensor = torch.tensor(tactile_points_quat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, len(tactile_points), 1)

        return tactile_points_pos_tensor, tactile_points_quat_tensor
    
    def get_tactile_points_in_world(self):
        num_tactile_points = self.tactile_pos_local.shape[1]

        pad_handle = self.rb_handles['pad_body_id_env']

        body_pos = self.body_pos[:, pad_handle].unsqueeze(1).expand(self.num_envs, num_tactile_points, 3)
        body_quat = self.body_quat[:, pad_handle].unsqueeze(1).expand(self.num_envs, num_tactile_points, 4)
        quat_list_per_env, pos_list_per_env = tf_combine(body_quat, body_pos,
                                                         self.tactile_quat_local, self.tactile_pos_local)

        return pos_list_per_env, quat_list_per_env

    def initialize_penalty_based_tactile(self, num_divs, sdf_tool='trimesh'):
        # initialize tactile sensing points
        self.tactile_pos_local, self.tactile_quat_local = self.generate_tactile_points(num_divs=num_divs)
        self.num_tactile_points = self.tactile_pos_local.shape[1]
        
        # initialize sdf
        self.sdf_tool = sdf_tool
        self.sdf = self.load_sdf_oracle_of_ball()
        
        # initialize coefficients
        self.tactile_kn = 1.
        self.tactile_damping = 0.003
        self.tactile_mu = 2.
        self.tactile_kt = 0.1
        
        self.visualize = False
        
    ### Trimesh
    def construct_sdf_with_trimesh(self, mesh):
        import trimesh
        proximity_query = trimesh.proximity.ProximityQuery(mesh)
        return proximity_query
    
    def query_distance_and_normal_with_trimesh(self, sdf, query_points):
        dtype = query_points.dtype
        query_points_np = query_points.cpu().numpy()
        
        distance_np = sdf.signed_distance(query_points_np)
        mask = (distance_np > 0.) # positive for penetration
        
        distance = torch.tensor(distance_np, dtype=dtype, device=self.device)
        normal = torch.zeros(query_points.shape, dtype=dtype, device=self.device)
        if mask.sum() > 0:
            closest_np, _, _ = sdf.on_surface(query_points_np[mask])
            
            closest = torch.tensor(closest_np, dtype=dtype, device=self.device)
            
            normal[mask] = tu.normalize(closest - query_points[mask])

            if self.visualize:
                import trimesh

                pointcloud_other_query = trimesh.PointCloud(query_points_np[~mask], colors=(0., 0., 1.))
                pointcloud_query = trimesh.PointCloud(query_points_np[mask], colors=(0., 1., 0.))
                pointcloud_closest = trimesh.PointCloud(closest_np, colors=(1., 0., 0.))
                
                # trimesh.Scene([self.ball_mesh, pointcloud_other_query, pointcloud_query, pointcloud_closest]).show()
                trimesh.Scene([pointcloud_other_query, pointcloud_query, pointcloud_closest]).show()
            
        return distance, normal
    
    ###  pysdf
    def construct_sdf_with_pysdf(self, mesh):
        from pysdf import SDF
        sdf = SDF(mesh.vertices, mesh.faces)
        return sdf

    def query_distance_and_normal_with_pysdf(self, sdf, query_points):
        dtype = query_points.dtype
        query_points_np = query_points.cpu().numpy()
        
        distance_np = sdf(query_points_np)
        mask = (distance_np > 0.) # positive for penetration
        
        distance = torch.tensor(distance_np, dtype = dtype, device = self.device)
        normal = torch.zeros(query_points.shape, dtype = dtype, device = self.device)
        
        if mask.sum() > 0:
            eps = 1e-5
            grad_x = sdf(query_points_np[mask] + np.array([eps, 0., 0.])) - sdf(query_points_np[mask] - np.array([eps, 0., 0.]))
            grad_y = sdf(query_points_np[mask] + np.array([0., eps, 0.])) - sdf(query_points_np[mask] - np.array([0., eps, 0.]))
            grad_z = sdf(query_points_np[mask] + np.array([0., 0., eps])) - sdf(query_points_np[mask] - np.array([0., 0., eps]))
            grad = torch.tensor(np.stack((grad_x, grad_y, grad_z), axis = 1), dtype = dtype, device = self.device) # check dimension
            normal[mask] = -tu.normalize(grad)

        return distance, normal
    
    ### PhysX SDF
    def query_distance_and_normal_with_physx(self, query_points):
        self.gym.refresh_sdf_view_tensor(self.sim,
                                         gymtorch.unwrap_tensor(self.sdf_shape_global_ids_per_env),
                                         gymtorch.unwrap_tensor(query_points))
        distance = -self.sdf_tensor[:, 0, :, 3]
        normal = self.sdf_tensor[:, 0, :, :3]
            
        return distance, normal
    
    #### Analytical
    def query_distance_and_normal_with_analytical(self, query_points):
        distance = self.ball_radius - query_points.norm(dim=-1)
        normal = tu.normalize(query_points)
            
        return distance, normal
    
    def load_sdf_oracle_of_ball(self):
        ball_sdf_asset_file = f'urdf/ball_rolling/ball_sdf.urdf'
        robot = URDF.load(os.path.join(self.asset_root, ball_sdf_asset_file))
        self.ball_mesh = robot.links[-1].visuals[0].geometry.mesh.meshes[0]
        origin = robot.links[-1].visuals[0].origin

        tf_mat = origin
        tf_pos = tf_mat[0:3, 3]
        tf_quat = R.from_matrix(tf_mat[0:3, 0:3]).as_quat()
        self.ball_mesh_tf = (torch.tensor(tf_quat, dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, 4),
                             torch.tensor(tf_pos, dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, 3))

        if self.sdf_tool == 'trimesh':
            sdf = self.construct_sdf_with_trimesh(self.ball_mesh)
        elif self.sdf_tool == 'pysdf':
            sdf = self.construct_sdf_with_pysdf(self.ball_mesh)
        elif self.sdf_tool == 'physx':
            # intialize the sdf object indices
            self.sdf_shape_global_ids_per_env = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
            ball_actor_handle = self.actor_handles['ball']
            for env_id, env_ptr in enumerate(self.env_ptrs):
                indenter_rb_id_actor = self.gym.find_actor_rigid_body_index(env_ptr, ball_actor_handle, 'ball', gymapi.DOMAIN_ACTOR)
                indenter_rb_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, ball_actor_handle)
                indenter_rb_shape_id_actor = indenter_rb_shape_indices[indenter_rb_id_actor].start
                indenter_rb_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_actor_handle)
                self.sdf_shape_global_ids_per_env[env_id, 0] = indenter_rb_shape_props[indenter_rb_shape_id_actor].global_index
            
            sdf = None
        elif self.sdf_tool == "analytical":
            self.ball_radius = 0.02
            sdf = None
        else:
            raise NotImplementedError
        
        return sdf
    
    '''
    Query collisions in sdf
    inputs:
        sdf: signed-distance field of the object
        tf_sdf: (pos, quat) of the object/sdf frame
        linvel_sdf, angvel_sdf: linear/angular velocity of the sdf object in world frame
        points_world, velocity_world: points and their velocities in world frame
    outputs:
        depth, depth_dot, normal, vt (normal and vt are in world frame)
    '''
    def query_collision(self, sdf, tf_sdf, sdf_linvel_world, sdf_angvel_world, points_world, velocity_world):
        dtype = points_world.dtype
        num_envs = self.num_envs
        num_points_per_env = self.num_tactile_points
        
        tf_sdf = (tf_sdf[0].unsqueeze(1).expand((num_envs, num_points_per_env, 4)), tf_sdf[1].unsqueeze(1).expand((num_envs, num_points_per_env, 3)))
        sdf_linvel_world = sdf_linvel_world.unsqueeze(1).expand((num_envs, num_points_per_env, 3))
        sdf_angvel_world = sdf_angvel_world.unsqueeze(1).expand((num_envs, num_points_per_env, 3))
        
        tf_sdf_inv = tu.tf_inverse(tf_sdf[0], tf_sdf[1])

        # compute points in the object frame
        points_sdf = tu.tf_apply(tf_sdf_inv[0], tf_sdf_inv[1], points_world)

        # compute depth
        if self.sdf_tool == 'trimesh':
            depth, normal_sdf = \
                self.query_distance_and_normal_with_trimesh(sdf, points_sdf.view(-1, 3))
            # collision_mask = collision_mask.view(num_envs, num_points_per_env)
            depth = depth.view(num_envs, num_points_per_env)
            normal_sdf = normal_sdf.view(num_envs, num_points_per_env, 3)
        elif self.sdf_tool == 'pysdf':
            depth, normal_sdf = \
                self.query_distance_and_normal_with_pysdf(sdf, points_sdf.view(-1, 3))
            # collision_mask = collision_mask.view(num_envs, num_points_per_env, )
            depth = depth.view(num_envs, num_points_per_env, )
            normal_sdf = normal_sdf.view(num_envs, num_points_per_env, 3)
        elif self.sdf_tool == 'physx':
            depth, normal_sdf = \
                self.query_distance_and_normal_with_physx(points_sdf.view(self.num_envs, 1, num_points_per_env, 3))
        elif self.sdf_tool == 'analytical':
            depth, normal_sdf = \
                self.query_distance_and_normal_with_analytical(points_sdf)
        else:
            raise NotImplementedError
                
        depth = depth.clamp(min=0., max=None)
            
        # compute other returned values
        normal_world = tu.quat_apply(tf_sdf[0], normal_sdf)

        '''
        x = R.T (xw - p)
        xdot = Rdot.T (xw - p) + R.T (xwdot - pdot)
                = R.T [w].T (xw - p) + R.T (xwdot - pdot)
                = R.T (-[w] (xw - p) + xwdot - pdot)
                = R.T ((xw - p) x [w] + xwdot - pdot)
        '''
        velocity_sdf = tu.quat_apply(tf_sdf_inv[0], 
                                        torch.cross(points_world - tf_sdf[1], sdf_angvel_world, dim = -1) +
                                        velocity_world - sdf_linvel_world)
                        
        '''
        ddot = dd/dx * dx/dt = n.T * xdot
        '''
        depth_dot = torch.sum(normal_sdf * velocity_sdf, dim = -1)
        
        '''
        xc_world = R * xc + p
        xcdot_world = [w] R xc + R xcdot + pdot
                    = [w] R xc + pdot
        '''
        closest_points_sdf = points_sdf + depth.unsqueeze(-1) * normal_sdf
        closest_points_velocity_world = torch.cross(sdf_angvel_world, tu.quat_apply(tf_sdf[0], closest_points_sdf), dim=-1) + sdf_linvel_world
        relative_velocity_world = velocity_world - closest_points_velocity_world

        vt_world = relative_velocity_world - normal_world * torch.sum(normal_world * relative_velocity_world, dim = -1, keepdim = True)
        
        return depth, depth_dot, normal_world, vt_world
    
    def get_penalty_based_tactile_forces(self):
        # acquire transform and velocities of two objects
        ball_body_tf = self.get_transform('ball_body_id_env') # ((num_envs, 4), (num_envs, 3))# NOTE [JIE]: check whether quaternion is normalized
        ball_sdf_tf = tf_combine(ball_body_tf[0], ball_body_tf[1], self.ball_mesh_tf[0], self.ball_mesh_tf[1])

        ball_sdf_angvel_world, ball_sdf_linvel_world = self.get_velocity('ball_body_id_env') # ((num_envs, 3), (num_envs, 3))
        
        pad_tf = self.get_transform('pad_body_id_env') # ((num_envs, 4), (num_envs, 3))
        pad_angvel_world, pad_linvel_world = self.get_velocity('pad_body_id_env') # ((num_envs, 3), (num_envs, 3))
        
        tactile_pos_world, tactile_quat_world = self.get_tactile_points_in_world()
        tactile_velocity_world = torch.cross(pad_angvel_world.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3)), 
                                             tu.quat_apply(pad_tf[0].unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 4)),
                                                           self.tactile_pos_local), dim=-1) \
                                 + pad_linvel_world.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3))
                                 
        depth, depth_dot, normal_world, vt_world = self.query_collision(self.sdf, ball_sdf_tf, ball_sdf_linvel_world, ball_sdf_angvel_world,
                                                   tactile_pos_world, tactile_velocity_world)

        # compute tactile forces in world frame
        # tf_sensor = self.get_transform(')
        '''compute contact force'''
        # mask = depth > 0.
        # if mask.sum() > 0:
        #     print(mask.sum())
        fc_norm = self.tactile_kn * depth #- self.tactile_damping * depth_dot * depth
        fc_world = fc_norm.unsqueeze(-1) * normal_world
        
        '''compute frictional force'''
        vt_norm = vt_world.norm(dim=-1)
        ft_static_norm = self.tactile_kt * vt_norm
        ft_dynamic_norm = self.tactile_mu * fc_norm
        ft_world = - torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)
        # ft_world = -ft_dynamic_norm.unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)
        '''net tactile force'''
        tactile_force_world = fc_world + ft_world
        
        '''tactile force in tactile frame'''
        quat_pad_inv = tu.quat_conjugate(pad_tf[0])
        tactile_force_pad = tu.quat_apply(quat_pad_inv.unsqueeze(1).expand(self.num_envs, self.num_tactile_points, 4), tactile_force_world)
        
        UnitX = torch.tensor([1., 0., 0.], device=self.device)
        UnitY = torch.tensor([0., 1., 0.], device=self.device)
        UnitZ = torch.tensor([0., 0., -1.], device=self.device)
        tactile_normal_axis = tu.quat_apply(self.tactile_quat_local, UnitZ.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        tactile_shear_x_axis = tu.quat_apply(self.tactile_quat_local, UnitX.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        tactile_shear_y_axis = tu.quat_apply(self.tactile_quat_local, UnitY.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        
        tactile_normal_force = -(tactile_normal_axis * tactile_force_pad).sum(-1)
        tactile_shear_force_x = (tactile_shear_x_axis * tactile_force_pad).sum(-1)
        tactile_shear_force_y = (tactile_shear_y_axis * tactile_force_pad).sum(-1)
        tactile_shear_force = torch.cat((tactile_shear_force_x.unsqueeze(-1), tactile_shear_force_y.unsqueeze(-1)), dim=-1)

        if (tactile_normal_force < 0.).sum() > 0:
            print('[Warning] negative tactile normal force')
            
        return depth, tactile_normal_force, tactile_shear_force
    
def parse_args():

    # Add custom arguments
    custom_parameters = [
        {'name': '--num_envs', 'type': int, 'default': 1, 'help': 'Number of environments to create'},
        {'name': '--use_compliant_dyn', 'action': 'store_true', 'help': 'When True, use compliant dynamics to allow soft interpenetration.'},
        {'name': '--sdf_tool', 'type': str, 'default': 'analytical', 'choices': ['trimesh', 'pysdf', 'physx', 'analytical'], 'help': 'which package to compute sdf'},
        {'name': '--num_tactile_rows', 'type': int, 'default': 20, 'help': '#rows of tactile marker points'},
        {'name': '--num_tactile_cols', 'type': int, 'default': 20, 'help': '#cols of tactile marker points'},
        {'name': '--disable_tactile', 'action': 'store_true', 'help': "whether to disable the tactile sensor computation"},
        {'name': '--render', 'action': 'store_true', 'help': "whether to render the simulation or not"},
        {'name': '--tactile_frequency', 'type': int, 'default': 5, 'help': 'frequency to compute tactile sensor image'},
        {'name': '--record', 'action': 'store_true', 'help': "whether to record the tactile force images"}
    ]
    
    args = gymutil.parse_arguments(
        headless=True,
        description='',
        custom_parameters=custom_parameters,
    )

    return args
