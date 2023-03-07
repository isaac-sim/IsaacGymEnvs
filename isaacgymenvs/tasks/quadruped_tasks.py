import abc 
import torch

from enum import Enum
from isaacgymenvs.utils.torch_jit_utils import *

def exp_neg_sq(x: torch.Tensor, alpha: float = 1):
    """ 
    Computes the function f(x) = exp(-alpha * x ** 2)
    """
    return torch.exp(- alpha * x ** 2)

class Task(Enum):
    Zero = 0
    TargetLocation = 1 # Move to a specified goal location
    TargetHeading = 2 # Turn to face a specified heading direction
    TargetVelocity = 3 # Achieve a specified velocity

def get_task_class_by_name(task_name: str):
    task = Task[task_name]
    if task == Task.TargetVelocity:
        return TargetVelocity
    else:
        raise RuntimeError(f"Unrecognized task {task_name}")

class AbstractTask(abc.ABC):
    """ Abstract base class for tasks """
    def __init__(self, cfg, num_envs, dtype, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.dtype = dtype
        self.device = device
        self.after_init()
        
        env_ids = to_torch(range(0, self.num_envs), dtype=torch.int64, device=self.device)
        self.reset(env_ids)

    def after_init(self):
        pass

    def reset(self, env_ids):
        """ Reset the task """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_observation_dim():
        """ Return the additional observation dimension required """
        pass

    @abc.abstractmethod
    def compute_reward(self):
        pass 

    @abc.abstractmethod
    def compute_observation(self):
        pass 

class TargetVelocity(AbstractTask): 

    def after_init(self):
        self.target_speed_lower = self.cfg["targetSpeedRange"]["lower"]
        self.target_speed_upper = self.cfg["targetSpeedRange"]["upper"]
        assert self.target_speed_lower <= self.target_speed_upper
        self.target_direction = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.target_speed = torch.zeros((self.num_envs, 1), dtype=self.dtype, device=self.device)

    @staticmethod
    def get_observation_dim():
        return 3 + 1 # directional unit vector, target speed 
    
    def reset(self, env_ids):
        """ Reset subset of commands """
        # Sample a standard Gaussian
        d = torch.randn_like(self.target_direction[env_ids])
        # Normalize it; the resulting unit vector is uniform on the hypersphere
        d = d / torch.norm(d, dim=-1, keepdim=True)
        # Sample a standard uniform 
        v = torch.rand_like(self.target_speed[env_ids])
        # Translate from [0,1] to [l, u]
        l, u = self.target_speed_lower, self.target_speed_upper
        v = (u - l) * v + l
        self.target_direction[env_ids] = d
        self.target_speed[env_ids] = v
    
    def compute_reward(self, root_states: torch.Tensor):
        """
        args:
            root_states: [N, 13] tensor of root states in world frame
        """
        root_vel = root_states[:, 7:10]
        target_vel = self.target_direction * self.target_speed
        dot_prod = torch.sum(root_vel * target_vel, dim=-1)
        return exp_neg_sq(dot_prod, alpha=self.cfg["velErrorScale"])

    def compute_observation(self, root_states: torch.Tensor):
        """
        args:
            root_states: [N, 13] tensor of root states in world frame
        """
        target_direction = self.target_direction
        target_speed = self.target_speed
        root_rot = root_states[:, 3:7]
        heading_rot = calc_heading_quat_inv(root_rot)
        target_direction_local = my_quat_rotate(heading_rot, target_direction)
        return torch.cat([target_direction_local, target_speed], dim=-1)

# TODO: Refactor this into a class
def compute_reward_target_location(root_states, target_pos):
    """
    args: 
        root_states - robot root states in world frame
        target_location - desired location in world frame 
    """
    root_pos = root_states[: ,:3]
    return torch.exp(exp_neg_sq(torch.norm(root_pos - target_pos))) 

def compute_observation_target_location(root_states, target_pos):
    root_pos = root_states[: ,:3]
    root_rot = root_states[:, 3:7]
    heading_rot = calc_heading_quat_inv(root_rot)
    goal_pos = target_pos - root_pos
    goal_pos_local = my_quat_rotate(goal_pos, heading_rot)
    return goal_pos_local