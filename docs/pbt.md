### Decentralized Population-Based Training with IsaacGymEnvs

#### Overview

Applications of evolutionary algorithms to reinforcement learning have
been popularized by publications such as [Capture the Flag](https://www.science.org/doi/full/10.1126/science.aau6249) by DeepMind.
Diverse populations of agents trained simultaneously can more efficiently explore the space of behaviors compared
to an equivalent amount of compute thrown at a single agent.

Typically Population-Based Training (PBT) is utilized in the context of multi-agent learning and self-play.
Agents trained with PBT in multi-agent environments exhibit more robust behaviors and are less prone to overfitting 
and can avoid collapse modes common in self-play training.
Recent results in environments such as [StarCraft II](https://www.nature.com/articles/s41586-019-1724-z.epdf?author_access_token=lZH3nqPYtWJXfDA10W0CNNRgN0jAjWel9jnR3ZoTv0PSZcPzJFGNAZhOlk4deBCKzKm70KfinloafEF1bCCXL6IIHHgKaDkaTkBcTEv7aT-wqDoG1VeO9-wO3GEoAMF9bAOt7mJ0RWQnRVMbyfgH9A%3D%3D)
show that PBT is instrumental in achieving human-level performance in these task.

Implementation in IsaacGymEnvs uses PBT with single-agent environments to solve hard manipulation problems
and find good sets of hyperparameters and hyperparameter schedules.

#### Algorithm

In PBT, instead of training a single agent we train a population of N agents.
Agents with a performance considerably worse than a population best are stopped, their policy weights are replaced
with those of better performing agents, and the training hyperparameters and reward-shaping coefficients are changed
before training is resumed.

A typical implementation of PBT relies on a single central orchestrator that monitors the processes and restarts them
as needed (i.e. this is the approach used by Ray & RLLIB).
An alternative approach is decentralized PBT. It requires fewer moving parts and is robust to failure of any single component
(i.e. due to hardware issue). In decentralized PBT each process monitors its own standing with respect to the population,
restarts itself as needed, etc.

IsaacGymEnvs implements decentralized PBT that relies on access to a shared part of filesystem available to all agents.
This is trivial when experiments are executed locally, or in a managed cluster environment
such as Slurm. In any other environment a mounted shared folder can be used, i.e. with SSHFS.

The algorithm proceeds as follows:
- each agent continues training for M timesteps after which it saves a checkpoint containing its policy weights and learning hyperparameters
- after checkpoint is saved, the agent compares its own performance to other agents in the population; the performance is only
compared to other agent's checkpoints corresponding to equal or smaller amount of collected experience
(i.e. agents don't compare themselves against versions of other agents that learned from more experience) 
- if the agent is not in bottom X% of the population, it continues training without any changes
- if the agent is in bottom X% of the population, but its performance is relatively close to the best agent it continues training
with mutated hyperparameters
- if the agent is in bottom X% of the population and its performance is significantly worse than that of the best agent,
its policy weights are replaced with weights of an agent randomly sampled from the top X% of the population, and its hyperparameters are mutated
before the training is resumed.

The algorithm implemented here is documented in details in the following RSS 2023 paper: https://arxiv.org/abs/2305.12127
(see also website https://sites.google.com/view/dexpbt)

#### PBT parameters and settings

(These are in pbt hydra configs and can be changed via command line)
- `pbt.interval_steps` - how often do we perform the PBT check and compare ourselves against other agents.
Typical values are in 10^6-10^8 range (10^7 by default). Larger values are recommended for harder tasks.
- `pbt.start_after`- start PBT checks after we trained for this many steps after experiment start or restart. Larger values allow
the population to accumulate some diversity.
- `pbt/mutation` - a Yaml file (Hydra config) for a mutation scheme. Specifies which hyperparameters should be mutated and how.

See more parameter documentation in pbt_default.yaml

#### Mutation

The mutation scheme is controlled by a Hydra config, such as the following:
```
task.env.fingertipDeltaRewScale: "mutate_float"
task.env.liftingRewScale: "mutate_float"
task.env.liftingBonus: "mutate_float"

train.params.config.reward_shaper.scale_value: "mutate_float"
train.params.config.learning_rate: "mutate_float"
train.params.config.grad_norm: "mutate_float"

train.params.config.e_clip: "mutate_eps_clip"
train.params.config.mini_epochs: "mutate_mini_epochs"
train.params.config.gamma: "mutate_discount"
```

Mutation scheme specifies hyperparameter names that could be passed via CLI and their corresponding mutation function.
Currently available mutation functions are defined in isaacgymenvs/pbt/mutation.py

A typical float parameter mutation function is trivial:

```
def mutate_float(x, change_min=1.1, change_max=1.5):
    perturb_amount = random.uniform(change_min, change_max)
    new_value = x / perturb_amount if random.random() < 0.5 else x * perturb_amount
    return new_value 
```

Some special parameters such as the discount factor require special mutation rules.

#### Target objective

In order to function, PBT needs a measure of _performance_ for individual agents.
By default, this is just agent's average reward in the environment.
If the reward is used as a target objective, PBT obviously can't be allowed to modify the reward shaping coefficient
and other hyperparameters that affect the reward calculation directly.

The environment can define a target objective different from default reward by adding a value `true_objective` to
the `info` dictionary returned by the step function, in IsaacGymEnvs this corresponds to:
`self.extras['true_objective'] = some_true_objective_value`

Using a separate true objective allows to optimize the reward function itself, so the overall
meta-optimization process can only care about the final goal of training, i.e. only the success rate in an object manipulation problem.
See allegro_kuka.py for example.

#### Running PBT experiments

A typical command line to start one training session in a PBT experiment looks something like this:

```
$ python -m isaacgymenvs.train seed=-1 train.params.config.max_frames=10000000000 headless=True pbt=pbt_default pbt.workspace=workspace_allegro_kuka pbt.interval_steps=20000000 
  pbt.start_after=100000000 pbt.initial_delay=200000000 pbt.replace_fraction_worst=0.3 pbt/mutation=allegro_kuka_mutation task=AllegroKukaLSTM task/env=reorientation pbt.num_policies=8 pbt.policy_idx=0
```

Note `pbt.policy_idx=0` - this will start the agent #0. For the full PBT experiment we will have to start agents `0 .. pbt.num_policies-1`.
We can do it manually by executing 8 command lines with `pbt.policy_idx=[0 .. 7]` while taking care 
of GPU placement in a multi-GPU system via manipulating CUDA_VISIBLE_DEVICES for each agent.

This process can be automated by the `launcher`
(originally implemented in [Sample Factory](www.samplefactory.dev),
find more information in the [launcher documentation](https://www.samplefactory.dev/04-experiments/experiment-launcher/))

_(Note that the use of the launcher is optional, and you can run PBT experiments without it.
For example, multiple scripts can be started in the computation medium of your choice via a custom shell script)._

##### Running PBT locally with multiple GPUs

The launcher uses Python scripts that define complex experiments. See `isaacgymenvs/experiments/allegro_kuka_reorientation_lstm_pbt.py` as an example.

This script defines a single experiment (the PBT run) with ParamGrid iterating over policy indices `0 .. num_policies-1`.
The experiment described by this script can be started on a local system using the following command:

```
python -m isaacgymenvs.pbt.launcher.run --run=isaacgymenvs.pbt.experiments.allegro_kuka_reorientation_pbt_lstm --backend=processes --max_parallel=8 --experiments_per_gpu=2 --num_gpus=4
```

On a 4-GPU system this will start 8 individual agents, fitting two on each GPU.

##### Running PBT locally on a single GPUs

```
python -m isaacgymenvs.pbt.launcher.run --run=isaacgymenvs.pbt.experiments.ant_pbt --backend=processes --max_parallel=4 --experiments_per_gpu=4 --num_gpus=1
```

##### Running PBT on your cluster

The launcher can be used to run PBT on the cluster. It currently supports local runners (shown above) and Slurm, though the Slurm cluster backend is not thoroughly tested with this codebase as of yet.

You can learn more about using the launcher to run on a Slurm cluster [here](https://www.samplefactory.dev/04-experiments/experiment-launcher/#slurm-backend)


##### Testing the best policy

The best checkpoint for the entire population can be found in <pbt_workspace_dir>/best<policy_idx> where <pbt_workspace_dir> is the shared folder, and policy_idx is 0,1,2,... It is decentralized so each policy saves a copy of what it thinks is the best versions from the entire population, but usually checking workspace/best0 is enough. The checkpoint name will contain the iteration index and the fitness value, and also the index of the policy that this checkpoint belongs to
