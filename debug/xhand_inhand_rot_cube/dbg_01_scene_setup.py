import os
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R
import time

# Initialize Gym
gym = gymapi.acquire_gym()

# Simulation Configuration
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 100.0
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.substeps = 1
# PhysX-specific parameters
# Ref: isaacgymenvs/tasks/factory/factory_schema_config_base.py
# sim_params.physx.use_gpu = True
# sim_params.use_gpu_pipeline = True
sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.001
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.use_gpu = True

# Create Simulator
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create simulator")

# Load Assets
asset_root = "./assets"
robot_asset_file = "urdf/xhand/xhand_right.urdf"
# robot_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
object_asset_file = "urdf/objects/cube_multicolor.urdf"

# Robot asset
robot_asset_options = gymapi.AssetOptions()
robot_asset_options.fix_base_link = True
# robot_asset_options.disable_gravity = True
# robot_asset_options.collapse_fixed_joints = True
# robot_asset_options.flip_visual_attachments = True
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, robot_asset_options)
if robot_asset is None:
    raise Exception("Failed to load robot asset")

# Object asset
object_asset = gym.load_asset(sim, asset_root, object_asset_file)
if object_asset is None:
    raise Exception("Failed to load object asset")

# Create Environment
num_envs = 4
envs = []
env_spacing = 2.0  # Space between environments
lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, int(num_envs ** 0.5))
    envs.append(env)

    # Add robot to environment
    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0, -0.1, 0.5)  # Position (x, y, z)
    qx, qy, qz, qw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat() # scalar-last by default
    # robot_pose.r = gymapi.Quat(0, 0, 0, 1)  # Rotation (x, y, z, w)
    robot_pose.r = gymapi.Quat(qx, qy, qz, qw)  # Rotation (x, y, z, w)
    robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", i, 0)

    # Add object to environment
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(0, 0, 0.7)  # Position (x, y, z)
    object_handle = gym.create_actor(env, object_asset, object_pose, "object", i, 0)
    
    # Add plane ground to environment
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # # pretty color
    # gym.set_rigid_body_color(
    #     env, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.9))
    
# Run the Simulation
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
view_step = 0   
while not gym.query_viewer_has_closed(viewer): # while True:
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    # gym.sync_frame_time(sim)
    
    if view_step == 0:
        time.sleep(3)
        view_step = 1


# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
