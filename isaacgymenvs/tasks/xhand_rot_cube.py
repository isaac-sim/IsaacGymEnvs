# Ref: isaacgymenvs/tasks/allegro_hand.py
# Ref: https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#task-logic
# - task setup

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymapi
from isaacgym import gymtorch

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
    
class XHandRotCube(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, 
                 virtual_screen_capture = False, force_render = False):
        
        # set config variable, so that it can by used by the task
        self.cfg = cfg
        
        # set the config params which not set up in the config file
        self.cfg["env"]["numActions"] = 12
        
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        # control parameters
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.object_actor_handles = []
        
        # observation and reward parameters
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        
        self.envs = []
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, 
                         virtual_screen_capture=virtual_screen_capture, 
                         force_render=force_render)
        
        self.hand_num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.hand_num_dofs)
        self.prev_targets = torch.zeros((self.num_envs, self.hand_num_dofs), dtype=torch.float, device=self.device)
        self.hand_default_dof_pos = torch.zeros(self.hand_num_dofs, dtype=torch.float, device=self.device)
        
        # set position of the camera to get a better view
        if self.viewer != None:
            cam_pos = gymapi.Vec3(7.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # shape: (num_action_in_all_envs, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.initial_root_state_tensor = self.root_state_tensor.clone()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
            
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, 
                                      self.graphics_device_id,
                                      self.physics_engine,
                                      self.sim_params)
        
        # create ground plane
        self._create_ground_plane()
        
        # create environments
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        # set world bounds
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # load robot and object assets
        asset_root = "./assets"
        robot_asset_file = "urdf/xhand/xhand_right.urdf"
        object_asset_file = "urdf/objects/cube_multicolor.urdf"
        goal_asset_file = "urdf/objects/cube_multicolor.urdf"
        
            
        ## set asset options
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.disable_gravity = True
        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.disable_gravity = True
        
        ## load asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_asset_options)
        if robot_asset is None:
            raise Exception("Failed to load robot asset")
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file)
        if object_asset is None:
            raise Exception("Failed to load object asset")

        goal_asset = self.gym.load_asset(self.sim, asset_root, goal_asset_file, goal_asset_options)
        if goal_asset is None:
            raise Exception("Failed to load goal object asset")

        self.object_indices = []
        self.target_indices = []
        self.hand_indices = []
        
        # create environment
        ## create env grid
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
        ## create actor
        ### create robot actor
            hand_init_pose = gymapi.Transform()
            hand_init_pose.p = gymapi.Vec3(0, -0.1, 0.5)
            qx, qy, qz, qw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat() # scalar-last by default
            hand_init_pose.r = gymapi.Quat(qx, qy, qz, qw)
            hand_actor = self.gym.create_actor(env_ptr, robot_asset, hand_init_pose, "robot_hand", i, -1)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
        ### create object actor
            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(0, 0, 0.8)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1)
            object_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_actor_handles.append(object_actor)
            
        ### create goal object actor
            self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
            goal_start_pose = gymapi.Transform()
            goal_start_pose.p = object_start_pose.p + self.goal_displacement
            goal_start_pose.p.z -= 0.04
            qx, qy, qz, qw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat() # scalar-last by default
            goal_start_pose.r = gymapi.Quat(qx, qy, qz, qw)
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.target_indices.append(goal_object_idx)
            # goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            # self.goal_object_indices.append(goal_object_idx)
        
        ## set dof properties
            self.num_hand_dofs = self.gym.get_asset_dof_count(robot_asset)
            self.actuated_dof_indices = [i for i in range(self.num_hand_dofs)]
            self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
            hand_dof_props = self.gym.get_asset_dof_properties(robot_asset)
            
            self.hand_dof_lower_limits = []
            self.hand_dof_upper_limits = []
            self.hand_dof_default_pos = []
            self.hand_dof_default_vel = []
            
            for i in range(self.num_hand_dofs):
                self.hand_dof_lower_limits.append(hand_dof_props['lower'][i])
                self.hand_dof_upper_limits.append(hand_dof_props['upper'][i])
                self.hand_dof_default_pos.append(0.0)
                self.hand_dof_default_vel.append(0.0)

                print("Max effort: ", hand_dof_props['effort'][i])
                hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                hand_dof_props['effort'][i] = 0.5
                hand_dof_props['stiffness'][i] = 3
                hand_dof_props['damping'][i] = 0.1
                hand_dof_props['friction'][i] = 0.01
                hand_dof_props['armature'][i] = 0.001
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)

        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)
        self.hand_dof_default_pos = to_torch(self.hand_dof_default_pos, device=self.device)
        self.hand_dof_default_vel = to_torch(self.hand_dof_default_vel, device=self.device)
        
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.target_indices = to_torch(self.target_indices, dtype=torch.long, device=self.device)


    def pre_physics_step(self, actions):
        # Ref: https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#pre-and-post-physics-step
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # print(f"reset_env_ids: {reset_env_ids}")
        
        # apply action
        self.actions = actions.clone().to(self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        
        self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])
        self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                    self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        
    
    def post_physics_step(self):
        # Ref: https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#pre-and-post-physics-step
        self.progress_buf += 1

        # env_ids = self.reset_buf.nonzero(
        #     as_tuple=False).squeeze(-1)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations(env_ids)
        self.compute_reward(env_ids)

    
    def reset_idx(self, env_ids):
        # Ref: isaacgymenvs/tasks/ant.py
        # reset object pose
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_state_tensor),
                                                     gymtorch.unwrap_tensor(torch.tensor(self.object_indices, dtype=torch.int32, device=self.device)), 
                                                     len(self.object_indices))
                
        # reset hand pose
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        
        pos = self.hand_default_dof_pos
        self.prev_targets[env_ids, :self.hand_num_dofs] = pos
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.prev_targets),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids_int32))

        # zero reset buffer
        self.reset_buf[env_ids_int32] = 0
    
    def compute_observations(self, env_idx):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        
        self.target_pose = self.root_state_tensor[self.target_indices, 0:7]
        self.target_pos = self.root_state_tensor[self.target_indices, 0:3]
        self.target_rot = self.root_state_tensor[self.target_indices, 3:7]
        
    
    def compute_reward(self, env_idx):
        # reset_buf is a tensor of shape (num_envs,) with boolean values, true if the environment should be reset
        self.reset_buf, self.rew_buf = compute_reward_fn(
            reset_buf=self.reset_buf, 
            object_pos=self.object_pos, target_pos=self.target_pos, fall_dist=self.fall_dist,
            object_rot=self.object_rot, target_rot=self.target_rot, rot_eps=self.rot_eps, rot_reward_scale=self.rot_reward_scale, dist_reward_scale=self.dist_reward_scale,
            actions=self.actions, action_penalty_scale=self.action_penalty_scale
            )
        pass
    
@torch.jit.script
def compute_reward_fn(reset_buf:torch.Tensor,
                      object_pos, target_pos, fall_dist:float,
                      object_rot, target_rot, rot_eps:float, rot_reward_scale:float, dist_reward_scale:float,
                      actions:torch.Tensor, action_penalty_scale:float,):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    # print("Goal dist: ", goal_dist)
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    # ignore_z_rot = False
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
    
    return resets, reward