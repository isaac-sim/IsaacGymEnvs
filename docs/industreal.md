# IndustRealSim

Here we provide extended documentation on IndustRealSim, which contains the environments and policy training code used in [Tang and Lin, et al., "IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality," Robotics: Science and Systems (RSS), 2023](https://arxiv.org/abs/2305.17110).

Before starting to use IndustRealSim, we would **highly** recommend familiarizing yourself with Isaac Gym, including the simpler RL examples. Optionally, you can also familiarize yourself with the [Factory examples](https://gitlab.com/nvidia_srl/factory/isaacgymenvs/-/blob/btang/main_rss/docs/factory.md), as the IndustRealSim examples have a similar code structure and reuse some classes and modules from Factory. 

<table align="center">
    <tr>
        <th>Initialization of Peg Insertion</th>
        <th>Trained Peg Insertion Policy</th>
        <th>Initialization of Gear Insertion</th>
        <th>Trained Gear Insertion Policy</th>
    </tr>
    <tr>
        <td><img src="https://github.com/bingjietang718/bingjietang718.github.io/assets/78517784/5d14452f-06ab-41cd-8545-bcf303dc4229" alt="drawing" width="200"/></th>
        <td><img src="https://github.com/bingjietang718/bingjietang718.github.io/assets/78517784/0baeaf2d-a21d-47e9-b74a-877ad59c4112" alt="drawing" width="200"/></th>
        <td><img src="https://github.com/bingjietang718/bingjietang718.github.io/assets/78517784/52df52f0-b122-4429-b6e2-b0b6ba9c29f6" alt="drawing" width="200"/></th>
        <td><img src="https://github.com/bingjietang718/bingjietang718.github.io/assets/78517784/af383243-3165-4255-9606-4a1419baee27" alt="drawing" width="200"/></th>
    </tr>
</table>

---

## Overview

There are 2 IndustRealSim example tasks: **IndustRealTaskPegsInsert** and **IndustRealTaskGearsInsert**. The first time you run these examples, it may take some time for Gym to generate signed distance field representations (SDFs) for the assets. However, these SDFs will then be cached.

**IndustRealTaskPegsInsert** and **IndustRealTaskGearsInsert** train policies for peg insertion tasks and gear insertion tasks, respectively. They correspond very closely to the code used to train the same policies in the IndustReal paper, but due to simplifications and improvements, may produce slightly different results than the original implementations.

---

## Running the Examples

1. Enter `isaacgymenvs/isaacgymenvs`
2. Train a policy (may take 8-10 hours for peg task and 18-20 hours for gear task to achieve high success rates on a modern GPU):
	* To train a policy **from scratch**:
`python train.py task=[task name]` (where `[task name]` is either `IndustRealTaskPegsInsert` or `IndustRealTaskGearsInsert`)
	* To train a policy **without rendering**:
`python train.py task=[task name] headless=True`
	* To resume policy training **from a specific checkpoint**:
 `python train.py task=[task name] checkpoint=[path to checkpoint]`
 3. Test the policy:
`python train.py task=[task name] checkpoint=[path to checkpoint] test=True`

---

## Best Practices

- If you modify the example code, run your code locally with rendering before you run it without rendering or on a cluster. Make sure that the pegs and gears are stable at the beginning of each episode (i.e., there are no initial explosions, the gripper closes properly on the pegs or gears, and there is no severe interpenetration between assets during contact). If you run into any simulation-related issues, go through the steps listed in the `Best Practices and Debugging` subsection of the [Factory documentation](https://gitlab.com/nvidia_srl/factory/isaacgymenvs/-/blob/btang/main_rss/docs/factory.md#collisions-and-contacts).

- If you run into a CUDA out-of-memory error, try the following:
	- Decrease `numEnvs` in `isaacgymenvs/isaacgymenvs/cfg/[task name].yaml`
	- Decrease `max_gpu_contact_pairs` or `default_buffer_size_multiplier` in [IndustRealBase.yaml](../isaacgymenvs/cfg/task/IndustRealBase.yaml)

---

## Core Code Details

### Classes

The core classes are the **IndustRealTaskPegsInsert** task class ([industreal_task_pegs_insert.py](../isaacgymenvs/tasks/industreal/industreal_task_pegs_insert.py)) and the **IndustRealTaskGearsInsert** task class ([industreal_task_gears_insert.py](../isaacgymenvs/tasks/industreal/industreal_task_gears_insert.py)). The core simulation-based policy training algorithms (i.e., **Simulation-Based Policy Update (SAPU)**, **SDF-Based Reward**, and **Sampling-Based Curriculum (SBC)**) are implemented in the IndustReal algorithms module ([industreal_algo_utils.py](../isaacgymenvs/tasks/industreal/industreal_algo_utils.py)). The update reward buffer method (`update_rew_buf()`) from the task classes calls functions from the algorithms module.

### Configuration Files

The core configuration files are the [IndustRealTaskPegsInsert.yaml](../isaacgymenvs/cfg/task/IndustRealTaskPegsInsert.yaml) and [IndustRealTaskGearsInsert.yaml](../isaacgymenvs/cfg/task/IndustRealTaskGearsInsert.yaml) task configuration files and the [IndustRealTaskPegsInsertPPO.yaml](../isaacgymenvs/cfg/train/IndustRealTaskPegsInsertPPO.yaml) and [IndustRealTaskGearsInsertPPO.yaml](../isaacgymenvs/cfg/train/IndustRealTaskGearsInsertPPO.yaml) training configuration files.

---

## Additional Code Details

These details may not be important unless you wish to significantly extend IndustRealSim.

### Classes

The class hierarchy for the IndustRealSim examples has a structure very similar to the class hierarchy from [Factory](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/factory.md). In summary, the **IndustRealTaskPegsInsert** task class ([industreal_task_pegs_insert.py]()) and **IndustRealTaskGearsInsert** task class ([industreal_task_gears_insert.py]()) inherit the **IndustRealEnvPegs** environment class ([industreal_task_env_pegs.py]()) and **IndustRealEnvGears** environment class ([industreal_task_env_gears.py]()), respectively. In turn, both environment classes inherit the **IndustRealBase** base class ([industreal_base.py]()). In addition, to minimize code duplication with Factory, the **IndustRealBase** class inherits the **FactoryBase** class ([factory_base.py](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/factory/factory_base.py)), and the IndustReal task classes use the Factory control module ([factory_control.py](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/factory/factory_control.py)).

### Configuration Files

In addition to the task and training configuration files described earlier, there are also base-level configuration files and environment-level configuration files. The base-level configuration file is [IndustRealBase.yaml](../isaacgymenvs/cfg/task/IndustRealBase.yaml), and the environment-level configuration files are [IndustRealEnvPegs.yaml](../isaacgymenvs/cfg/task/IndustRealEnvPegs.yaml) and [IndustRealEnvGears.yaml](../isaacgymenvs/cfg/task/IndustRealEnvGears.yaml).

### Schema

There are abstract base classes that define the necessary methods for base, environment, and task classes ([factory_schema_class_base.py](../isaacgymenvs/tasks/factory/factory_schema_class_base.py), [factory_schema_class_env.py](../isaacgymenvs/tasks/factory/factory_schema_class_env.py), and [factory_schema_class_task.py](../isaacgymenvs/tasks/factory/factory_schema_class_task.py)). These are useful to review in order to better understand the structure of the code, but you will probably not need to modify them. They are also recommended to inherit if you would like to add your own environments and tasks.

There are also schema for the base-level, environment-level, and task-level configuration files ([factory_schema_config_base.py](../isaacgymenvs/tasks/factory/factory_schema_config_base.py), [factory_schema_config_env.py](../isaacgymenvs/tasks/factory/factory_schema_config_env.py), and [factory_schema_config_task.py](../isaacgymenvs/tasks/factory/factory_schema_config_tasks.py)). These schema are enforced for the base-level and environment-level configuration files, but not for the task-level configuration files. These are useful to review in order to better understand the structure of the configuration files and see descriptions of common parameters, but again, you will probably not need to modify them.

### Franka URDF

As described in Section V.C of the [IndustReal paper](https://arxiv.org/pdf/2305.17110.pdf), in order to facilitate sim-to-real transfer, arbitrary dissipative terms were removed from asset descriptions, which includes the default Franka URDF. In addition, the Franka URDF was compared to the official specification sheet, and a number of small corrections and modifications were made. For more details, please see the [URDF file](../assets/industreal/urdf/industreal_franka.urdf) as well as the informal [changelog](../assets/industreal/urdf/industreal_franka_urdf_changelog.txt).

---

## Frequently Asked Questions

1. How are pegs/gears initialized?

We first randomize socket/shaft poses on the tabletop and set the peg/gear to be on top of the socket/shaft. Then, we move the gripper to the peg/gear grasp pose. When the gripper approaches the peg/gear, it might collide with the peg/gear and cause the peg/gear to move. (You might see this when rendering the scene; it is completely normal.) We then reset the peg/gear to be on top of the socket/shaft again, so that at the beginning of each episode, the peg/gear will be grasped by the gripper when it closes. We did not move the gripper first and then introduce the peg/gear to avoid collision during initialization, because in Gym, all actors in each environment must be added at the very beginning of simulation.

2. How many simulation steps are needed for moving the arm and closing the gripper during initialization?

Since there is no joint limit on robot joints, when we move the arm end-effector to a certain pose and close the gripper, you may notice the robot's elbow drifts even when the end-effector is at its target pose. This can be effectively avoided by tuning the number of simulation steps for arm movements or by regularizing the elbow angle. 

3. Is RL policy training seed-dependent?

In IndustReal, we train with 5 random seeds with all experiments and select the one with the highest success rate. There might be certain seeds that take a longer time to train, but they should have similar performance after convergence.

---


## Citation
If you use any of the IndustRealSim training environments or algorithms in your work, please cite [IndustReal](https://arxiv.org/abs/2305.17110):
```
@inproceedings{
	tang2023industreal,
	author = {Bingjie Tang and Michael A Lin and Iretiayo Akinola and Ankur Handa and Gaurav S Sukhatme and Fabio Ramos and Dieter Fox and Yashraj Narang},
	title = {IndustReal: Transferring contact-rich assembly tasks from simulation to reality},
	booktitle = {Robotics: Science and Systems},
	year = {2023}
}
```

The simulation methods, original environments, and low-level control algorithms were described in our preceding work, [Factory](https://arxiv.org/abs/2205.03532), which you may want to refer to or cite as well:
```
@inproceedings{
	narang2022factory,
	author = {Yashraj Narang and Kier Storey and Iretiayo Akinola and Miles Macklin and Philipp Reist and Lukasz Wawrzyniak and Yunrong Guo and Adam Moravanszky and Gavriel State and Michelle Lu and Ankur Handa and Dieter Fox},
	title = {Factory: Fast contact for robotic assembly},
	booktitle = {Robotics: Science and Systems},
	year = {2022}
} 
```
