from isaacgymenvs.pbt.launcher.run_description import ParamGrid, RunDescription, Experiment
from isaacgymenvs.pbt.experiments.run_utils import version, seeds, default_num_frames


kuka_env = 'allegro_kuka_two_arms_reorientation'
_frames = default_num_frames

_name = f'{kuka_env}_{version}'

_params = ParamGrid([
    ('seed', seeds(8)),
])

_wandb_activate = True
_wandb_group = f'pbt_{_name}'
_wandb_entity = 'your_wandb_entity'
_wandb_project = 'your_wandb_project'

cli = f'python -m isaacgymenvs.train ' \
      f'train.params.config.max_frames={_frames} headless=True ' \
      f'task=AllegroKukaTwoArmsLSTM task/env=reorientation ' \
      f'wandb_project={_wandb_project} wandb_entity={_wandb_entity} wandb_activate={_wandb_activate} wandb_group={_wandb_group}'

RUN_DESCRIPTION = RunDescription(
    f'{_name}',
    experiments=[Experiment(f'{_name}', cli, _params.generate_params(randomize=False))],
    experiment_arg_name='experiment', experiment_dir_arg_name='hydra.run.dir',
    param_prefix='', customize_experiment_name=False,
)
