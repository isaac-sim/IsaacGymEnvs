# Ref: isaacgymenvs/tasks/allegro_hand.py
# Ref: https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#task-logic
# - task setup

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymapi
import torch

class XHandRotCube(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, 
                 virtual_screen_capture = False, force_render = False):
        
        # set config variable, so that it can by used by the task
        self.cfg = cfg
        
        # set the config params which not set up in the config file
        self.cfg["env"]["numActions"] = 12
        
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, 
                         virtual_screen_capture=virtual_screen_capture, 
                         force_render=force_render)

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, 
                                      self.graphics_device_id,
                                      self.physics_engine,
                                      self.sim_params)
        
        # create ground plane
        self._create_ground_plane()
        
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
        asset_file = "urdf/xhand/xhand_right.urdf"
        
            
        ## set asset options
        
        ## load asset
        
        # create environment
        ## create env grid
        ## create actor
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