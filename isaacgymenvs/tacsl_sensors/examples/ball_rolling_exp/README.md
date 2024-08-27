## Ball Rolling Experiment with Tactile Simulation

### Run the Code
```
python run.py --use_compliant_dyn --render
```
There are a few arguments that can be played with.
- `--num_envs`: determine the number of parallel environments to create
- `--sdf_tool`: determine which sdf tool to use, can be chosen from [analytical, physx, trimesh, pysdf]
- `--num_tactile_rows`, `--num_tactile_cols`: the resolution of the tactile point array
- `--tactile_frequency`: the frequency to compute tactile readings, by default the tactile information will be computed every 5 simulation steps.