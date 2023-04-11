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

import torch 

import isaacgymenvs.learning.amp_players as amp_players
from isaacgymenvs.utilities.tensor_history import TensorHistory, TensorIO
from typing import Dict, List, Tuple

class QuadrupedAMPPlayerContinuous(amp_players.AMPPlayerContinuous):

    def __init__(self, params):
        super().__init__(params)

        # Initialize loggers
        # TODO: find some way to avoid hardcoding these
        self.dt = 0.02
        self.num_envs = 1
        self.max_episode_len = 2000
        # TODO: find some way to avoid hardcoding tensor shapes
        self.tensors: List[Tuple[str, int]] = [
            ("root_states", 13), 
            ("dof_pos", 12),
            ("dof_vel", 12), 
            ("obs", 42), 
            ("prev_action", 12), 
            ("task_state", 11)
        ]
        self.tensor_histories: Dict[str, TensorHistory] = {}
        self.tensor_ios: Dict[str, TensorIO] = {}
        file_handle = TensorIO.new_file('dataset.h5')
        for name, tensor_dim in self.tensors:
            self.tensor_histories[name] = TensorHistory(
                max_len = self.max_episode_len, 
                tensor_shape = (self.num_envs, tensor_dim,),
                device = self.device
            )
            self.tensor_ios[name] = TensorIO(
                file_handle, 
                (self.max_episode_len, tensor_dim,), 
                name
            )
        file_handle.attrs['dt'] = self.dt 
        file_handle.attrs['max_episode_length'] = self.max_episode_len
        return

    def _post_step(self, info):
        super()._post_step(info)
        self._update_tensor_histories(info)
        return
    
    def _env_reset_done(self):
        obs, env_ids = super()._env_reset_done()
        if len(env_ids) > 0:
            # Assume all envs are reset simultaneously
            # This will be true when enableEarlyTermination=False
            assert len(env_ids) == self.num_envs
            self._save_tensor_histories()
            self._reset_tensor_histories()
        return obs, env_ids

    def _update_tensor_histories(self, info):
        for name, tensor_history in self.tensor_histories.items():
            assert name in info
            tensor_history.update(info[name])
        pass

    def _save_tensor_histories(self):
        if self.tensor_histories['root_states'].current_idx == 0:
            return
        for name, tensor_history in self.tensor_histories.items():
            th = tensor_history.get_history() # (T, N, d)    
            th = torch.transpose(th, 0, 1) # (N, T, d)
            tensor_io = self.tensor_ios[name]
            tensor_io.write(th.detach().cpu().numpy())

    def _reset_tensor_histories(self):
        for tensor_history in self.tensor_histories.values():
            tensor_history.clear()