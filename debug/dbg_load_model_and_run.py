# Ref: https://github.com/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_continuous.ipynb
# pip install envpool # https://pypi.org/project/envpool/

import yaml

params = """
params:


  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: False
    space:
      continuous:
        mu_activation: None
        sigma_activation: None

        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: True
    mlp:
      units: [128, 64, 32]
      activation: elu
      
      initializer:
        name: default
      regularizer:
        name: None
        
"""
        
# Load yaml from string as dictironary
config = yaml.safe_load(params)
print(config)
# # Load YAML as a dictionary
# with open('isaacgymenvs/cfg/train/BallBalancePPO.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# Print or use the dictionary
print(config)

from rl_games.torch_runner import Runner
config = {'params': {'algo': {'name': 'a2c_continuous'},
  'config': {'bound_loss_type': 'regularisation',
   'bounds_loss_coef': 0.0,
   'clip_value': False,
   'critic_coef': 4,
   'e_clip': 0.2,
   'entropy_coef': 0.0,
   'env_config': {'env_name': 'Pendulum-v1', 'seed': 5},
   'env_name': 'envpool',
   'full_experiment_name' : 'pendulum_onnx',
   'save_best_after' : 20,
   'gamma': 0.99,
   'grad_norm': 1.0,
   'horizon_length': 32,
   'kl_threshold': 0.008,
   'learning_rate': '3e-4',
   'lr_schedule': 'adaptive',
   'max_epochs': 200,
   'mini_epochs': 5,
   'minibatch_size': 1024,
   'name': 'pendulum',
   'normalize_advantage': True,
   'normalize_input': True,
   'normalize_value': True,
   'num_actors': 64,
   'player': {'render': False},
   'ppo': True,
   'reward_shaper': {'scale_value': 0.1},
   'schedule_type': 'standard',
   'score_to_win': 20000,
   'tau': 0.95,
   'truncate_grads': True,
   'use_smooth_clamp': False,
   'value_bootstrap': True},
  'model': {'name': 'continuous_a2c_logstd'},
  'network': {'mlp': {'activation': 'elu',
    'initializer': {'name': 'default'},
    'units': [32, 32]},
   'name': 'actor_critic',
   'separate': False,
   'space': {'continuous': {'fixed_sigma': True,
     'mu_activation': 'None',
     'mu_init': {'name': 'default'},
     'sigma_activation': 'None',
     'sigma_init': {'name': 'const_initializer', 'val': 0}}}},
  'seed': 5}}
runner = Runner()
runner.load(config)
runner.run({
    'train': True,
    # 'play': True,
})

agent = runner.create_player()
agent.restore('isaacgymenvs/runs/BallBalance_09-21-06-05/nn/BallBalance.pth')

import rl_games.algos_torch.flatten as flatten
import torch
inputs = {
    'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
    'rnn_states' : agent.states,
}
