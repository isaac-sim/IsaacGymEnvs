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
        
        self.progress_buf = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        env_ids = to_torch(range(0, self.num_envs), dtype=torch.int64, device=self.device)
        self.reset(env_ids)

    def after_init(self):
        pass

    def reset(self, env_ids):
        """ Reset the task """
        self.progress_buf[:] = 0

    def on_step(self):
        self.progress_buf += 1

    def get_state(self):
        return None
    
    def update_sim_state(self, root_states):
        self.root_states = root_states

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
        self.target_direction_reset = self.cfg["reset"]["targetDirection"]
        self.target_speed_reset = self.cfg["reset"]["targetSpeed"]
        self.target_yaw_rate_reset = self.cfg["reset"]["targetYawRate"]

        self.target_direction = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.target_speed = torch.zeros((self.num_envs, 1), dtype=self.dtype, device=self.device)
        self.target_yaw_rate = torch.zeros((self.num_envs, 1), dtype=self.dtype, device=self.device)

        self.use_schedule = self.cfg["reset"]["schedule"]["enabled"]
        if self.use_schedule:
            target_velocity_schedule = np.load(self.cfg["reset"]["schedule"]["path"])
            self.target_velocity_schedule = to_torch(target_velocity_schedule, device=self.device, dtype=self.dtype)
        
        self.use_position_pd = self.cfg['reset']["position_pd_control"]["enabled"]
        if self.use_position_pd:
            self.kp = self.cfg['reset']["position_pd_control"]["kP"]
            self.kd = self.cfg['reset']["position_pd_control"]["kD"]
            target_position_schedule = np.load(self.cfg['reset']["position_pd_control"]["path"])
            self.target_position_schedule = to_torch(target_position_schedule, device=self.device, dtype=self.dtype)
    
    def get_state(self):
        return torch.cat([
            self.target_direction, 
            self.target_speed, 
            self.target_yaw_rate,
            self.get_current_position(), 
            self.get_goal_position()
        ], dim=-1)

    def get_current_position(self):
        return self.root_states[:, :3]
    
    def get_current_velocity(self):
        return self.root_states[:, 7:10]
    
    def get_goal_position(self):
        if self.use_position_pd:
            return self.target_position_schedule[self.progress_buf, :]
        else:
            return torch.zeros_like(self.get_current_position())
    
    def get_goal_velocity(self):
        return torch.zeros_like(self.get_current_velocity())
    
    def compute_target_velocity(self):
        current_position = self.get_current_position()
        goal_position = self.get_goal_position()
        current_velocity = self.get_current_velocity()
        goal_velocity = self.get_goal_velocity()
        target_velocity = self.kd * (goal_position - current_position) + self.kp * (goal_velocity - current_velocity)
        target_velocity[:,2] = 0
        
        # Clip to [-2, 2]
        target_velocity = torch.clip(target_velocity, -1, 1)
        return target_velocity

    def on_step(self):
        super().on_step()
        if self.use_schedule:
            self.target_direction[:] = self.target_velocity_schedule[self.progress_buf, :3]
            self.target_speed[:] = self.target_velocity_schedule[self.progress_buf, 3:4]
        if self.use_position_pd:
            target_velocity = self.compute_target_velocity()
            self.target_direction[:] = target_velocity / torch.norm(target_velocity, dim=-1, keepdim=True)
            self.target_speed[:] = torch.norm(target_velocity, dim=-1, keepdim=True)
        

    def reset(self, env_ids):
        """ Reset subset of commands """
        tar_dir_rs = self.target_direction_reset["strategy"]
        if tar_dir_rs == "RandomUnitVector":
            self.reset_random_direction(env_ids)
        elif tar_dir_rs == "Fixed":
            self.reset_fixed_direction(env_ids)
        else: 
            raise RuntimeError(f"Unrecognized reset strategy {tar_dir_rs}")
        
        tar_spe_rs = self.target_speed_reset["strategy"]
        if tar_spe_rs == "RandomUniform":
            self.reset_random_speed(env_ids)
        else:
            raise RuntimeError(f"Unrecognized reset strategy {tar_spe_rs}")
        
        tar_dyaw_rs = self.target_yaw_rate_reset["strategy"]
        if tar_dyaw_rs == "Fixed":
            self.reset_fixed_yaw_rate(env_ids)
        elif tar_dyaw_rs == "RandomUniform":
            self.reset_random_yaw_rate(env_ids)
        else:
            raise RuntimeError(f"Unrecognized reset strategy {tar_dyaw_rs}")

    def reset_fixed_direction(self, env_ids):
        """ Set all direction vectors to fixed value """
        d = torch.zeros_like(self.target_direction[env_ids])
        val = to_torch(self.target_direction_reset["value"])
        d = torch.unsqueeze(val, 0)
        self.target_direction[env_ids] = d

    def reset_random_direction(self, env_ids):
        # Sample a standard Gaussian
        d = torch.randn_like(self.target_direction[env_ids])

        # We must have at least one axis
        axis = self.target_direction_reset["axis"]
        assert 1 <= len(axis) <= 3
        assert set(char for char in axis).issubset(set(char for char in 'xyz'))
        if 'z' not in self.target_direction_reset["axis"]:
            # Zero out the z-axis
            d[:, 2] = 0
        # Normalize it; the resulting unit vector is uniform on the hypersphere
        d = d / torch.norm(d, dim=-1, keepdim=True)
        self.target_direction[env_ids] = d

    def reset_random_speed(self, env_ids):
        """ Set speed to random speed """
        # Sample a standard uniform 
        v = torch.rand_like(self.target_speed[env_ids])
        # Translate from [0,1] to [l, u]
        l, u = self.target_speed_reset["lower"], self.target_speed_reset["upper"]
        v = (u - l) * v + l
        self.target_speed[env_ids] = v

    def reset_fixed_yaw_rate(self, env_ids):
        val = to_torch(self.target_yaw_rate_reset["value"])
        dyaw = torch.unsqueeze(val, 0)
        self.target_yaw_rate_reset[env_ids] = dyaw

    def reset_random_yaw_rate(self, env_ids):
        # Sample a standard uniform 
        dyaw = torch.rand_like(self.target_yaw_rate[env_ids])
        # Translate from [0,1] to [l, u]
        l, u = self.target_yaw_rate_reset["lower"], self.target_yaw_rate_reset["upper"]
        dyaw = (u - l) * dyaw + l
        self.target_speed[env_ids] = dyaw
 
    def compute_reward(self, root_states: torch.Tensor):
        """
        args:
            root_states: [N, 13] tensor of root states in world frame
        """
        root_vel = root_states[:, 7:10]
        dot_prod = torch.sum(root_vel * self.target_direction, dim=-1)
        target_speed = torch.squeeze(self.target_speed)
        linvel_scale = self.cfg["reward"]["linearVelocity"]["scale"]
        linvel_weight = self.cfg["reward"]["linearVelocity"]["weight"]
        return linvel_weight * exp_neg_sq((target_speed - dot_prod), alpha=linvel_scale)

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
        target_yaw_rate = self.target_yaw_rate
        return torch.cat([target_direction_local, target_speed, target_yaw_rate], dim=-1)
    
    @staticmethod
    def get_observation_dim():
        return 3 + 1 + 1 # directional unit vector, target speed, target yaw rate 

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