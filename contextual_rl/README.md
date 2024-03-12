# Contextual RL on Isaac Gym
This is a repository for contextual reinforcement learning (RL) on Isaac Gym.


## Installation
Make sure that you have installed the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym) and followed the instructions in the documentation.

Install this repo, which includes two locomotion environments (Ant, Anymal) modified to simulate environmental changes.
```
git clone https://github.com/CheongWoong/Contextual_IsaacGymEnvs
cd Contextual_IsaacGymEnvs/
pip install -e .
```


## Experiments

### Training
Run the following scripts to train models (noncontextual_baselines, contextual_baselines and privileged distillation, respectively).  
The model checkpoints and tensorboard logs are saved in 'contextual_rl/runs/training/seed_{training_seed}/{env_id}.'  
Modify the hyperparameters and gpu_id for your purpose.
```
# task_name: ['ant', 'anymal']
bash scripts/training/noncontextual_baselines/train_{task_name}.sh
bash scripts/training/contextual_baselines/train_{task_name}.sh
bash scripts/training/privileged_distilation/train_{task_name}.sh
```

### Evaluation
Run the following scripts to evaluate the last checkpoints of models (noncontextual_baselines, contextual_baselines and privileged distillation, respectively).  
The tensorboard logs are saved in 'contetxual_rl/runs/test/seed_{training_seed}/{test_env_id}/{training_env_id}/{checkpoint_idx}.
```
# task_name: ['ant', 'anymal']
bash scripts/test/noncontextual_baselines/test_{task_name}_last_checkpoint.sh
bash scripts/test/contextual_baselines/test_{task_name}_last_checkpoint.sh
bash scripts/test/privileged_distilation/test_{task_name}_last_checkpoint.sh
```

### Analysis (Work in progress)
Run the following scripts to collect the context embeddings from the test environments.  
```
```

Then, use the [ipython notebook](https://github.com/CheongWoong/Contextual_IsaacGymEnvs/tree/main/contextual_rl/analysis/context_embedding_analysis.ipynb) for visualization and analysis.
