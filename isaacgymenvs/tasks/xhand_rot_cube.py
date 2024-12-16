# Ref: isaacgymenvs/tasks/allegro_hand.py
# Ref: https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#task-logic
# - task setup

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymapi
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class XHandRotCube(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, 
                 virtual_screen_capture = False, force_render = False):
        
        # set config variable, so that it can by used by the task
        self.cfg = cfg
        
        # set the config params which not set up in the config file
        self.cfg["env"]["numActions"] = 12
        
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        self.envs = []
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, 
                         virtual_screen_capture=virtual_screen_capture, 
                         force_render=force_render)
        
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
        
            
        ## set asset options
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.disable_gravity = True
        
        ## load asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_asset_options)
        if robot_asset is None:
            raise Exception("Failed to load robot asset")
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file)
        if object_asset is None:
            raise Exception("Failed to load object asset")

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
            object_init_pose = gymapi.Transform()
            object_init_pose.p = gymapi.Vec3(0, 0, 0.6)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_init_pose, "object", i, -1)

        ## set dof properties
        
    
    def pre_physics_step(self, actions):
        pass
    
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