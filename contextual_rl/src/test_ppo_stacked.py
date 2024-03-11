# reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py

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

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import os
import random
import time
from dataclasses import dataclass

import gym
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from collections import deque


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 100
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Ant"
    """the id of the environment"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    record_video_step_frequency: int = 100
    """the frequency at which to record the videos"""
    device_id: int = 7 # cwkang: set the gpu id
    """the gpu id"""

    len_history: int = 10
    # cwkang: Checkpoint path to load the context encoder
    checkpoint_path: str = ""
    """the path to the checkpoint"""

    # to be filled in runtime
    total_episodes: int = 1024 # cwkang: this value will be set the same as num_envs
    """total episodes for evaluation"""
    

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


NUM_SYS_PARAMS = 2 # cwkang: add input dim
class Agent(nn.Module):
    def __init__(self, envs, len_history):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.context_encoder = nn.Sequential(
            layer_init(nn.Linear((obs_dim+action_dim)*len_history, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 10)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + 10, 256)), # cwkang: add input dim
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + 10, 256)), # cwkang: add input dim
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, history, x):
        context = self.context_encoder(history)
        return self.critic(torch.cat((context, x), dim=-1))

    def get_action_and_value(self, history, x, action=None):
        context = self.context_encoder(history)
        action_mean = self.actor_mean(torch.cat((context, x), dim=-1))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(torch.cat((context, x), dim=-1))
    
    def get_context(self, history):
        with torch.no_grad():
            return self.context_encoder(history)


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.total_episodes = args.num_envs
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}" # cwkang: use datetime format for readability
    checkpoint_idx=os.path.basename(args.checkpoint_path).replace('.pth', '') # cwkang: add filename_suffix for tensorboard summarywriter
    seed_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint_path))))
    source_env_id = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint_path)))
    run_name = f"test/{seed_id}/{os.path.join(args.env_id, source_env_id, checkpoint_idx)}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.env_id,
        num_envs=args.num_envs,
        sim_device=f"cuda:{args.device_id}" if torch.cuda.is_available() and args.cuda else "cpu",
        rl_device=f"cuda:{args.device_id}" if torch.cuda.is_available() and args.cuda else "cpu",
        graphics_device_id=0 if torch.cuda.is_available() and args.cuda else -1,
        headless=False if torch.cuda.is_available() and args.cuda else True,
        multi_gpu=False,
        virtual_screen_capture=args.capture_video,
        force_render=False,
    )
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.len_history).to(device)
    agent.load_state_dict(torch.load(f'{args.checkpoint_path}'))
    agent.eval()

    history_obs = deque(maxlen=args.len_history)
    history_action = deque(maxlen=args.len_history)
    history_done = deque(maxlen=args.len_history)

    for _ in range(args.len_history):
        history_obs.append(torch.zeros((args.num_envs,) + envs.single_observation_space.shape, dtype=torch.float).to(device))
        history_action.append(torch.zeros((args.num_envs,) + envs.single_action_space.shape, dtype=torch.float).to(device))
        history_done.append(torch.zeros((args.num_envs,), dtype=torch.float).to(device))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    sys_param_weight = envs.get_sys_param_weight() # cwkang: get system parameters

    from collections import defaultdict
    test_results = defaultdict(dict) # cwkang: record test results
    num_episodes = 0

    while num_episodes < args.total_episodes:
        global_step += args.num_envs

        # cwkang: store history obs and dones
        history_obs.append(next_obs)
        history_done.append(next_done)

        # ALGO LOGIC: action logic
        with torch.no_grad():
            #######
            # action, logprob, _, value = agent.get_action_and_value(next_obs)
                
            history_dones = torch.stack(list(history_done), dim=1)
            history_obses = torch.stack(list(history_obs), dim=1)
            history_actions = torch.stack(list(history_action), dim=1)

            last_done_indices = (history_dones == 1).cumsum(dim=1).max(dim=1).indices
            timesteps = torch.arange(history_dones.size(1), device=device).expand_as(history_dones)
            history_input_mask = timesteps >= last_done_indices.unsqueeze(-1)

            history_input_obs = history_obses*history_input_mask.unsqueeze(-1)
            history_input_action = history_actions*history_input_mask.unsqueeze(-1)
            history_input = torch.cat((history_input_obs, history_input_action), dim=-1)

            history_input = history_input.reshape((history_input.shape[0], -1))
            action, logprob, _, value = agent.get_action_and_value(history_input, next_obs)
            #######

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, info = envs.step(action)
        
        for idx, d in enumerate(next_done):
            if d:
                if idx in test_results['episodic_return']: # cwkang: one environment produces one result for fair comparison (evaluation is done with the same initial states)
                    continue

                episodic_return = info["r"][idx].item()
                episodic_length = info["l"][idx].item()
                test_results['episodic_return'][idx] = episodic_return # cwkang: record results
                test_results['episodic_length'][idx] = episodic_length # cwkang: record results

                if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                    consecutive_successes = info["consecutive_successes"].item()
                    test_results['consecutive_successes'][idx] = consecutive_successes # cwkang: record results

                num_episodes = len(test_results['episodic_return']) # cwkang: count the number of episodes for recording
                if num_episodes % (args.total_episodes // 10) == 0 or num_episodes == args.total_episodes:
                    print(f"{num_episodes} episodes done")

                if num_episodes == args.total_episodes:
                    break

        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(f"evaluation_seed_{args.seed}/SPS", int(global_step / (time.time() - start_time)), global_step)

    for key in test_results:
        for idx in sorted(list(test_results[key].keys())):
            writer.add_scalar(f"evaluation_seed_{args.seed}/{key}", test_results[key][idx], idx)

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print()
    for key in test_results:
        test_results[key] = list(test_results[key].values())
    print(f"episodic_return: mean={np.mean(test_results['episodic_return'])}, std={np.std(test_results['episodic_return'])}")
    print(f"episodic_length: mean={np.mean(test_results['episodic_length'])}, std={np.std(test_results['episodic_length'])}")
    writer.add_scalar(f"evaluation_seed_{args.seed}/episodic_return_mean", np.mean(test_results['episodic_return']), checkpoint_idx)
    writer.add_scalar(f"evaluation_seed_{args.seed}/episodic_return_std", np.std(test_results['episodic_return']), checkpoint_idx)
    writer.add_scalar(f"evaluation_seed_{args.seed}/episodic_length_mean", np.mean(test_results['episodic_length']), checkpoint_idx)
    writer.add_scalar(f"evaluation_seed_{args.seed}/episodic_length_std", np.std(test_results['episodic_length']), checkpoint_idx)
    if 'consecutive_successes' in test_results:
        print(f"consecutive_successes: mean={np.mean(test_results['consecutive_successes'])}, std={np.std(test_results['consecutive_successes'])}")
        writer.add_scalar(f"evaluation_seed_{args.seed}/consecutive_successes_mean", np.mean(test_results['consecutive_successes']), checkpoint_idx)
        writer.add_scalar(f"evaluation_seed_{args.seed}/consecutive_successes_std", np.std(test_results['consecutive_successes']), checkpoint_idx)


    # envs.close()
    writer.close()