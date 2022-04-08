from hydra import compose, initialize
from omegaconf import OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator


OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def make(task: str, num_envs: int):
    initialize(config_path="./cfg")
    cfg = compose(config_name="config", overrides=[f"task={task}"])
    cfg_dict = omegaconf_to_dict(cfg.task)
    cfg_dict['env']['numEnvs'] = num_envs

    create_rlgpu_env = get_rlgames_env_creator(
        cfg_dict,
        task,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        multi_gpu=False,
    )
    return create_rlgpu_env()