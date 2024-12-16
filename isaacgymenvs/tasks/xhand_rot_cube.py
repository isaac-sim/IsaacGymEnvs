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
        
        self.envs = []
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, 
                         virtual_screen_capture=virtual_screen_capture, 
                         force_render=force_render)
        
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        # set position of the camera to get a better view
        if self.viewer != None:
            cam_pos = gymapi.Vec3(7.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
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
        ### create object actor
            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(0, 0, 0.6)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1)
        ### create goal object actor
            self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
            goal_start_pose = gymapi.Transform()
            goal_start_pose.p = object_start_pose.p + self.goal_displacement
            goal_start_pose.p.z -= 0.04
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
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
        
    def pre_physics_step(self, actions):
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
        pass
    
    def reset_idx(self, env_idx):
        pass
    
    def compute_observation(self, env_idx):
        pass
    
    def compute_reward(self, env_idx):
        pass
    
@torch.jit.script
def compute_reward_fn():
    pass