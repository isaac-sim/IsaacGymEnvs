from isaacgymenvs.pbt.launcher.run_description import ParamGrid, RunDescription, Experiment
from isaacgymenvs.pbt.experiments.run_utils import version


_env = 'ant'
_name = f'{_env}_{version}'
_iterations = 10000
_pbt_num_policies = 3

_params = ParamGrid([
    ('pbt.policy_idx', list(range(_pbt_num_policies))),
])

_wandb_activate = True
_wandb_group = f'pbt_{_name}'
_wandb_entity = 'your_wandb_entity'
_wandb_project = 'your_wandb_project'

_experiments = [
    Experiment(
        f'{_name}',
        f'python -m isaacgymenvs.train task=Ant headless=True '
        f'max_iterations={_iterations} num_envs=2048 seed=-1 train.params.config.save_frequency=2000 '
        f'wandb_activate={_wandb_activate} wandb_group={_wandb_group} wandb_entity={_wandb_entity} wandb_project={_wandb_project} '
        f'pbt=pbt_default pbt.num_policies={_pbt_num_policies} pbt.workspace=workspace_{_name} '
        f'pbt.initial_delay=10000000 pbt.interval_steps=5000000 pbt.start_after=10000000 pbt/mutation=ant_mutation',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription(
    f'{_name}',
    experiments=_experiments, experiment_arg_name='experiment', experiment_dir_arg_name='hydra.run.dir',
    param_prefix='', customize_experiment_name=False,
)
