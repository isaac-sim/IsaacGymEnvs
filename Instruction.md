# 环境配置
先安装rlgpu环境，官方链接https://developer.nvidia.com/isaac-gym
保证环境能 运行 python/examples/joint_monkey.py
clone GIT仓库https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git到本地

安装环境 pip install -e .
# 文件结构
assert 放置urdf文件
task 为主要任务代码，包括环境新建，reward函数，观察state。
cfg 配置文件
runs 下存放日志文件，以及训练权重
# 开始训练
训练xarm机械臂执行命令
python train.py task=xarmCubeStack


# 推理命令

task=xarmCubeStack test=True num_envs=32 checkpoint=/home/tany/GITHUB/IsaacGymEnvs/isaacgymenvs/runs/FrankaCubeStack_29-10-53-28/nn/last_FrankaCubeStack_ep_800_rew_10.961887.pth sim_device=cpu
checkpoint 为模型权重位置。
