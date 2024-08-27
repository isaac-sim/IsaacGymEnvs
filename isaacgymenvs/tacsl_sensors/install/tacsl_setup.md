## TacSL: A Library for Visuo-tactile Sensor Simulation and Learning


🔴 **The official TacSL code will be released in the new [Isaac Lab](https://isaac-sim.github.io/IsaacLab/index.html) framework with the Isaac Simulator.** 🔴

Momentarily, the TacSL policy learning toolkit is available for the deprecated Isaac Gym simulator.
This code is simply to reproduce the results in the paper and to offer a preview of what to expect in IsaacLab.
Please note that we do not plan to provide further support for this version.


---

## Set up
To use TacSL policy learning module within the **deprecated** Isaac Gym simulator, here are the instructions:
- Create a python 3.8 environment e.g. with a conda virtual env.
```commandline
conda create --name tacsl python==3.8
```

- Download TacSL-specific Isaac Gym binary from [here](https://drive.google.com/file/d/1FHs1tf3QajvYb11UkLaLcDD9THL-C0G5/view?usp=sharing)
and pip install within python 3.8:
```commandline
pip install -e IsaacGym_Preview_TacSL_Package/isaacgym/python/
```

- Install the `tacsl` branch of IsaacGymEnvs from [here](../../..):
```commandline
pip install -e isaacgymenvs
```

- Install the latest [rl_games](https://github.com/Denys88/rl_games) from source:
```commandline
git clone https://github.com/Denys88/rl_games.git
pip install -e ./rl_games
```

- Install additional dependencies from the [requirements.txt](./requirements.txt):
```commandline
pip install -r requirements.txt
```
- Download the Gelsight assets from [here](https://drive.google.com/file/d/1kf-F4RdHdKiNZpNLi-fSV-KE0ny72_L0/view?usp=sharing) and place them in the IGE folder [here](../../../assets/tacsl/mesh)

---
## Running

- To confirm that installing IsaacGym is successful, try running the simple examples e.g.:
```commandline
cd IsaacGym_Preview_TacSL_Package/isaacgym/python/examples
python franka_osc.py
```

- Policy learning: to train a state-based policy, run:
```commandline
cd isaacgymenvs/tacsl_sensors/install
./train_state.sh
```
  - See [here](../../../docs/tacsl.md)
    for more detailed documentation and instructions on training policies with different sensor modalities.


---

## Frequently Asked Questions

1) If you encounter error running IsaacGym: `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`

```commandline
export LD_LIBRARY_PATH=/home/${USER}/anaconda3/envs/tacsl/lib:$LD_LIBRARY_PATH
```

2) For python-related error, confirm that the packages in the [requirements.txt](./requirements.txt) are installed with the specified versions.
