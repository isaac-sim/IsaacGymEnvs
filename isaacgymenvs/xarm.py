
import yaml
import isaacgym
from isaacgymenvs.utils.utils import set_seed

# 假设你的环境类在这个模块中
from tasks import xarmCubeStack
import torch
# 加载环境配置
config_path = '/home/tany/GITHUB/IsaacGymEnvs/isaacgymenvs/cfg/task/xarmCubeStacktest.yaml'
with open(config_path, 'r') as file:
    cfg = yaml.safe_load(file)
def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


# 环境初始化参数
rl_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
graphics_device_id = 0
headless = False

# 设置种子，可以选择任何整数值
seed = 1234
set_seed(seed)

# 创建环境实例
env = xarmCubeStack(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False,
                    force_render=False)

# 运行仿真循环
num_steps = 1000000
for _ in range(num_steps):
    # 生成随机动作

    actions = torch.rand((env.num_envs, cfg["env"]["numActions"]), device=rl_device)
    # 执行动作并渲染环境

    env.step(actions)

    env.render()

# 关闭环境
env.close()