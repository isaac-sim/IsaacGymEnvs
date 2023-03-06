# Copyright (c) 2018-2022, NVIDIA Corporation
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

import numpy as np
import os
import torch

from gym import spaces
from enum import Enum
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.quadruped_amp_base import QuadrupedAMPBase, compute_quadruped_observations
from isaacgymenvs.utilities.quadruped_motion_data import MotionLib

from typing import Tuple, Dict

def random_uniform(n: int, lower: torch.Tensor, upper: torch.Tensor, device):
    return torch.unsqueeze(upper - lower, 0) * torch.rand((n, ) + upper.shape, device=device)

def random_uniform_quaternion(n: int, device) -> torch.Tensor:
    """
    Reference: Top answer to https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
    """
    two_pi = np.pi * 2
    u = torch.zeros(n).uniform_(0., 1)
    v = torch.zeros(n).uniform_(0., 1)
    w = torch.zeros(n).uniform_(0., 1)

    qx = torch.sqrt(1-u) * torch.sin(two_pi * v)
    qy = torch.sqrt(1-u) * torch.cos(two_pi * v)
    qz = torch.sqrt(u) * torch.sin(two_pi * w)
    qw = torch.sqrt(u) * torch.cos(two_pi * w)
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return q.to(device)

class QuadrupedAMP(QuadrupedAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3
        RandomPose = 4

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # AMP-specific
        state_init = cfg["env"]["stateInit"]
        self._state_init = QuadrupedAMP.StateInit[state_init]
        self._enable_ref_state_init_height = cfg["env"]["enableRefStateInitHeight"]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        self._num_amp_obs_per_step = 3 + 4 + 3 + 3 + 12 + 12 # root pos, root orn, root lin vel, root ang vel, dof pos, dof vel 
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._local_root_obs = cfg["env"]["localRootObs"]

        super().__init__(cfg=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Load motion file
        self._motion_file = cfg['env']['motionFile']
        self._load_motion(self._motion_file)

        # Initialize _amp_obs_buf, _curr_amp_obs_buf, _hist_amp_obs_buf, _amp_obs_demo_buf (?)
        self._amp_obs_space = spaces.Box(np.ones(self.get_num_amp_obs()) * -np.Inf, np.ones(self.get_num_amp_obs()) * np.Inf)
        
        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = compute_quadruped_observations(root_states, dof_pos, dof_vel,
                                      self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat
    
    def _build_amp_obs_demo_buf(self, num_samples):
        """ Builds the AMP observation buffer. 
        
        Only needs to be called at init. """
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _load_motion(self, motion_file):
        """ Loads a motion library to do AMP training"""
        self._motion_lib = MotionLib(motion_file, self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == QuadrupedAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == QuadrupedAMP.StateInit.RandomPose):
            self._reset_random_pose(env_ids)
        elif (self._state_init == QuadrupedAMP.StateInit.Start
              or self._state_init == QuadrupedAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == QuadrupedAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        
        return
    
    def _reset_default(self, env_ids):
        """ Default initialization of robot. 
        
        Robot is initialized from default initial state. """
        # TODO: Replace 
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = self.default_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        """ Reference state initialization of robot. 
        
        For each env, a reference motion is selected and used to initialize the robot state."""
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == QuadrupedAMP.StateInit.Random
            or self._state_init == QuadrupedAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == QuadrupedAMP.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        if self._enable_ref_state_init_height:
            # Set root (x,y) pos to begin at same position
            root_pos[:,:2] = self.initial_root_states[env_ids,:2]
        else:
            # Additionally set z-pos to begin at task.env.baseInitState.pos
            root_pos[:,:3] = self.initial_root_states[env_ids,:3]

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_random_pose(self, env_ids):
        """ Reset to a random starting pose """
        # TODO: Undo hardcoding of various constants
        num_envs = len(env_ids)
        dof_pos = random_uniform(num_envs, self.dof_limits_lower, self.dof_limits_upper, device=self.device)
        dof_vel = torch.zeros_like(self.default_dof_vel[env_ids]).uniform_(-0.2, 0.2) # m/s
        root_pos = self.initial_root_states[env_ids,:3].clone()
        root_pos[:,2] = torch.zeros_like(self.initial_root_states[env_ids,2]).uniform_(0.6, 1.2) # m
        root_rot = random_uniform_quaternion(num_envs, device=self.device)
        root_vel = torch.zeros_like(self.initial_root_states[env_ids,7:10]).uniform_(-0.1, 0.1)
        root_ang_vel = torch.zeros_like(self.initial_root_states[env_ids,10:13]).uniform_(-0.1, 0.1)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)


    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        random_pose_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(random_pose_reset_ids) > 0):
            self._reset_random_pose(random_pose_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        """ Default initialization of AMP obs
        
        AMP obs kept the same as previously. """
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        """ Reference state initialization of AMP obs
        
        Overwrite the amp_obs demonstration with the sampled motion IDs and times"""
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = compute_quadruped_observations(root_states, dof_pos, dof_vel,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel
        
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        """ Update history of AMP obs by shifting the timestep forward by 1. """
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = compute_quadruped_observations(self.root_states, self.dof_pos, self.dof_vel,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = compute_quadruped_observations(self.root_states[env_ids], self.dof_pos[env_ids], 
                                                                    self.dof_vel[env_ids],
                                                                    self._local_root_obs)
        return