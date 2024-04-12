# 环境配置
先安装rlgpu环境，官方链接https://developer.nvidia.com/isaac-gym

官方给的torch版本为1.8 在运行某些task的时候回报错 我改成了1.13

```bash
conda create -n rlgpu python=3.7

conda activate rlgpu
```
````
  - python=3.7
  - pytorch=1.13.1+cu117
  - torchvision=0.14.1+cu117
  - pyyaml>=5.3.1
  - scipy>=1.5.0
  - tensorboard=2.11.2
````

```bash
cd python
pip install -e .
```

保证环境能 运行 
```bash
python python/examples/joint_monkey.py
```

clone GIT仓库[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git)到本地

在原来rlgpu的环境下


```bash
cd IsaacGymEnvs
```


安装环境 
```bash
pip install -e .
```




# 文件结构
`assert` 放置urdf文件

`task文件夹下的任务类` 为主要任务代码，包括环境新建，reward函数，观察state。

`cfg` 配置文件

`runs` 下存放日志文件，以及训练权重.查看训练日志 tensorboard --logdir=runs

# 开始训练
训练xarm机械臂执行命令

```bash
python train.py task=xarmCubeStack num_envs=1024
```


# 推理命令
```bash
task=xarmCubeStack test=True num_envs=32 checkpoint=/home/tany/GITHUB/IsaacGymEnvs/isaacgymenvs/runs/FrankaCubeStack_29-10-53-28/nn/last_FrankaCubeStack_ep_800_rew_10.961887.pth sim_device=cpu

```

checkpoint 为模型权重位置。
