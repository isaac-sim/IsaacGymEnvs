

# AutoMate

Here we provide extended documentation on AutoMate, which contains the environments and policy training code used in [Tang, et al., "AutoMate: Specialist and Generalist Assembly Policies over Diverse Geometries," ](https://bingjietang718.github.io/automate/) Robotics: Science and Systems (RSS), 2024.

Before starting to use AutoMate, we would **highly** recommend familiarizing yourself with Isaac Gym, including the simpler RL examples. Optionally, you can also familiarize yourself with the [Factory examples](factory.md) and  [IndustReal examples](industreal.md), as the AutoMate examples have a similar code structure and reuse some classes and modules from Factory and IndustReal. 


## Overview

There are 3 AutoMate example tasks: **AutoMateTaskAsset**, **AutoMateTaskDisassemble** and **AutoMateTaskAssemble**. The first time you run these examples, it may take some time for Gym to generate signed distance field representations (SDFs) for the assets. However, these SDFs will then be cached.

 - **AutoMateTaskAsset** provides empty simulation environments where the assemblies are loaded without any RL training code. 
 - **AutoMateTaskDisassemble** collects disassembly paths for assemblies by first loading the assemblies in their assembled state, moving the robot to grasp the plug, and lifting the plug.
 - **AutoMateTaskAssemble** trains specialist policies for assembling parts included in our dataset. They correspond very closely to the code used to train the same policies in the AutoMate paper, but due to simplifications and improvements, may produce slightly different results than the original implementations.


## Dataset

We provide a dataset of 100 assemblies compatible with simulation and 3D printable in the real world (this dataset is derived from [Assemble Them All](https://assembly.csail.mit.edu/)). Mesh files of our dataset are stored in [this directory](../assets/automate). For each assembly in our dataset, we show its unique ID and a rendering.

![asset_lookup_table](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/assets/78517784/80c37c51-8776-4518-977d-659510fecd57)


## Running the Examples

 - Enter `isaacgymenvs/isaacgymenvs`
 - Run the example:
	 - **Load assemblies** without any RL training code:
	```bash
	python train.py task=AutoMateTaskAsset
	```
	 - **Load a specified list of assemblies** (e.g., 00004, 00007) without any RL training code: 
		- In [AutoMateTaskAsset.yaml](../isaacgymenvs/cfg/task/AutoMateTaskAsset.yaml), set `task.env.desired_subassemblies: ['asset_00004', 'asset_00007']`.
		- Then, in command line:
	```bash
	python train.py task=AutoMateTaskAsset task.env.overwrite_subassemblies=True 
	```
	 - **Collect disassembly paths** for a given assembly (e.g., 00346), the collected paths will be saved [here](../isaacgymenvs/tasks/automate/data/): 
	```bash
	python train.py task=AutoMateTaskDisassemble task.env.overwrite_subassemblies=True task.env.desired_subassemblies=['asset_00346']
	```
	 - **Train an assembly policy** for a given assembly (e.g., 00346): 
		- **NOTE**: To train an assembly policy, disassembly paths need to be collected in advance.
	```bash
	python train.py task=AutoMateTaskAssemble task.env.overwrite_subassemblies=True task.env.desired_subassemblies=['asset_00346']
	```
- **NOTE**: The first time you run these examples, it may take a long time (~30 minutes) for Gym to generate signed distance field representations (SDFs) for the assets. However, these SDFs will then be cached (i.e., the next time you run the same examples, it will load the previously generated SDFs).
- **NOTE**: AutoMateTaskDisassemble only supports disassembly path collection for one assembly and AutoMateTaskAssemble only supports policy training for one assembly at the moment.
- Other useful command line arguments:
	 - To run the examples **without rendering**, add: `headless=True`
	 - To resume training from a specific **checkpoint**, add: `checkpoint=[path to checkpoint]`
	 - To **test** a trained policy, add: `checkpoint=[path to trained policy checkpoint] test=True`
	 - To change the number of parallelized environments, add: `task.env.numEnvs=[number of environments]` 
	 - To set a random seed for RL training, add: `seed=-1`, to set a specific seed, add `seed=[seed number]`
	 - To set maximum number of iterations for RL training, add: `max_iterations=[number of max RL training iterations]`
	 - To test a policy, add: `test=True task.env.if_eval=True checkpoint=[path to trained policy checkpoint]`

---

## Core Code Details

The **core task files** are: 
 - For *AutoMateTaskAsset* task (no RL training): 
   - Class file: [automate_task_asset.py](../isaacgymenvs/tasks/automate/automate_task_asset.py)
   - Task configuration file: [AutoMateTaskAsset.yaml](../isaacgymenvs/cfg/task/AutoMateTaskAsset.yaml) 
 - For *AutoMateTaskDisassemble* task (no RL training): 
   - Class file: [automate_task_disassemble.py](../isaacgymenvs/tasks/automate/automate_task_disassemble.py)
   - Task configuration file: [AutoMateTaskDisassemble.yaml](../isaacgymenvs/cfg/task/AutoMateTaskDisassemble.yaml) 
 - For *AutoMateTaskAssemble* task: 
   - Class file: [automate_task_assemble.py](../isaacgymenvs/tasks/automate/automate_task_assemble.py)
   - Task configuration file: [AutoMateTaskAssemble.yaml](../isaacgymenvs/cfg/task/AutoMateTaskAssemble.yaml)  
   - Training configuration file: [AutoMateTaskAssemblePPO.yaml](../isaacgymenvs/cfg/train/AutoMateTaskAssemblePPO.yaml)

The **core simulation-based policy training algorithms** (i.e., RL with an imitation objective, trajectory matching via dynamic time warping) are implemented in the AutoMate algorithms module ([automate_algo_utils.py](../isaacgymenvs/tasks/automate/automate_algo_utils.py)). 

Other **related data files** are:
 - [plug_grasps.json](../isaacgymenvs/tasks/automate/data/plug_grasps.json), which stores the grasp poses for each plug used in AutoMate. Grasp samples for each plug are generated with [this library](https://github.com/NVlabs/DefGraspSim/tree/main/graspsampling-py-defgraspsim).
 - [disassembly_dist.json](../isaacgymenvs/tasks/automate/data/disassembly_dist.json), which stores the lift distance for each assembly (i.e., the distance to disassemble each plug from the corresponding socket). We compute this distance by first generating convex hulls of the plug mesh and the socket mesh, loading them in their assembled state, gradually moving the plug convex hull in +z direction until the two convex hulls no longer intersect with each other, and logging the distance the plug has traveled along z-axis. We use [Warp](https://github.com/NVIDIA/warp) to compute the distance for all assemblies. 

## Citation
If you use any of the AutoMate dataset, training environments or algorithms in your work, please cite [AutoMate](https://bingjietang718.github.io/automate/):
```
@inproceedings{
	tang2024automate,
	author    = {Tang, Bingjie and Akinola, Iretiayo and Xu, Jie and Wen, Bowen and Handa, Ankur and Van Wyk, Karl and Fox, Dieter and Sukhatme, Gaurav S. and Ramos, Fabio and Narang, Yashraj},
	title     = {AutoMate: Specialist and Generalist Assembly Policies over Diverse Geometries},
	booktitle = {Robotics: Science and Systems},
	year      = {2024},
}
```

The simulation-based policy training methods also include algorithms described in our previous work [IndustReal](https://arxiv.org/abs/2305.17110), which you may want to refer to or cite as well:
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

