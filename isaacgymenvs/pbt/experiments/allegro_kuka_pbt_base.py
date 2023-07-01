from isaacgymenvs.pbt.launcher.run_description import ParamGrid, RunDescription, Experiment
from isaacgymenvs.pbt.experiments.run_utils import version, default_num_frames


kuka_env = 'allegro_kuka'
_frames = default_num_frames

_pbt_num_policies = 8
_name = f'{kuka_env}_{version}_pbt_{_pbt_num_policies}p'

_wandb_activate = True
_wandb_group = f'pbt_{_name}'
_wandb_entity = 'your_wandb_entity'
_wandb_project = 'your_wandb_project'

kuka_base_cli = (f'python -m isaacgymenvs.train seed=-1 '
                 f'train.params.config.max_frames={_frames} headless=True '
                 f'wandb_project={_wandb_project} wandb_entity={_wandb_entity} wandb_activate={_wandb_activate} wandb_group={_wandb_group} '
                 f'pbt=pbt_default pbt.workspace=workspace_{kuka_env} '
                 f'pbt.interval_steps=20000000 pbt.start_after=100000000 pbt.initial_delay=200000000 pbt.replace_fraction_worst=0.3 pbt/mutation=allegro_kuka_mutation')

_params = ParamGrid([
    ('pbt.policy_idx', list(range(_pbt_num_policies))),
])

cli = kuka_base_cli + f' task=AllegroKuka task/env=reorientation pbt.num_policies={_pbt_num_policies}'

RUN_DESCRIPTION = RunDescription(
    f'{_name}',
    experiments=[Experiment(f'{_name}', cli, _params.generate_params(randomize=False))],
    experiment_arg_name='experiment', experiment_dir_arg_name='hydra.run.dir',
    param_prefix='', customize_experiment_name=False,
)
