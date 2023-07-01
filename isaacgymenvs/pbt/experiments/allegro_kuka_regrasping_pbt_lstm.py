from isaacgymenvs.pbt.launcher.run_description import ParamGrid, RunDescription, Experiment
from isaacgymenvs.pbt.experiments.allegro_kuka_pbt_base import kuka_env, kuka_base_cli
from isaacgymenvs.pbt.experiments.run_utils import version


_pbt_num_policies = 8
_name = f'{kuka_env}_regrasp_{version}_pbt_{_pbt_num_policies}p'
_wandb_group = f'pbt_{_name}'

_params = ParamGrid([
    ('pbt.policy_idx', list(range(_pbt_num_policies))),
])

cli = kuka_base_cli + f' task=AllegroKukaLSTM task/env=regrasping wandb_activate=True wandb_group={_wandb_group} pbt.num_policies={_pbt_num_policies}'

RUN_DESCRIPTION = RunDescription(
    f'{_name}',
    experiments=[Experiment(f'{_name}', cli, _params.generate_params(randomize=False))],
    experiment_arg_name='experiment', experiment_dir_arg_name='hydra.run.dir',
    param_prefix='', customize_experiment_name=False,
)
