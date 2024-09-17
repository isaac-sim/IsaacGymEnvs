import numpy as np 
import torch
from gym import spaces
from isaacgymenvs.tasks.base.vec_task import VecTask

class PrivInfoVecTask(VecTask):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, **kwargs): 
        """Initialise the `PrivInfoVecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless, **kwargs)
        self.config = config
        self._allocate_task_buffer()
        self.obs_space = spaces.Dict(
            {
                "obs": spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf),
                "priv_info": spaces.Box(np.ones(self.num_env_factors) * -np.Inf, np.ones(self.num_env_factors) * np.Inf),
                "proprio_hist": spaces.Box(np.ones(self.prop_hist_len) * -np.Inf, np.ones(self.prop_hist_len) * np.Inf),
            }
        )

 
        # TODO: populate proprio_hist_buf history buf at each timestpe
        # https://github.com/HaozhiQi/hora/blob/main/hora/tasks/allegro_hand_hora.py

        # TODO: on tdmpc end, optional encoder for priv_info / proprio_hist from config 
        # TODO: on tdmpc end, batch/num_envs tensor shapes

    def _allocate_task_buffer(self):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['propHistoryLen']
        self.num_env_factors = self.config['env']['privInfoDim']
        self.priv_info_buf = torch.zeros((self.num_envs, self.num_env_factors), device=self.device, dtype=torch.float)

        # TODO: fix 32 below 
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)
        
        # print(f"Priviliged Information Buffer Dimension: {self.num_env_factors.shape} <- (num_envs, priv_info_dim)")
        # print(f"Proprioceptiion Buffer Dimension: {self.proprio_hist_buf.shape} <- (num_envs, num_env_factors, num_env_factors)")

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        # breakpoint()
        return self.obs_dict

    def step(self, actions):
       super().step(actions)
       self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
       self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
       return self.obs_dict, self.rew_buf, self.reset_buf, self.extras
       
        
