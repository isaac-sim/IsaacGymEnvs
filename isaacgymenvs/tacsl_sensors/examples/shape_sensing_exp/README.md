## Shape Sensing Experiment with Tactile Simulation (RGB and Force-Field)

### Run the Code


- Gelsight indenter demo
  - Gelsight R1.5
```commandline
python tactile_viz_rgb_and_shear.py --num_envs 1  --compliance_stiffness 200 --use_tactile_rgb --use_tactile_ff --render

python tactile_viz_rgb_and_shear.py --num_envs 1  --compliance_stiffness 200 --use_tactile_rgb --use_tactile_ff --indenter_name bounding_sphere
```
  - Gelsight Mini
```commandline
python tactile_viz_rgb_and_shear.py --num_envs 1 --sensor_type gs_mini --use_tactile_rgb --compliance_stiffness 300
```

There are a few arguments that can be played with.
- `--num_envs`: determine the number of parallel environments to create
- `--sdf_tool`: determine which sdf tool to use, can be chosen from [analytical, physx, trimesh, pysdf]
- `--num_tactile_rows`, `--num_tactile_cols`: the resolution of the tactile point array.
- `--render`: render the simulation environment to screen


- Calibration weights experiment:

```commandline
python tactile_viz_rgb_and_shear.py --num_envs 1  --compliance_stiffness 7.5   --use_tactile_rgb --render --distance_to_sensor 0.01 --indenter_name weight_20g --floating_indenter
```
