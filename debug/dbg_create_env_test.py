import os
from isaacgym import gymapi

# Initialize Gym
gym = gymapi.acquire_gym()

# Simulation Configuration
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.up_axis = gymapi.UP_AXIS_Z

# Create Simulator
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create simulator")

# Load Assets
asset_root = "." #"path/to/assets"
robot_asset_file = "isaacgymenvs/balance_bot.xml" #"urdf/robot.urdf"
object_asset_file ="assets/urdf/ball.urdf" #"models/object.obj"

# Robot asset
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file)
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
    robot_pose.p = gymapi.Vec3(0, 0, 1)  # Position (x, y, z)
    robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", i, 1)

    # Add object to environment
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(0.5, 0, 1.5)  # Position (x, y, z)
    object_handle = gym.create_actor(env, object_asset, object_pose, "object", i, 1)
    
    # Add plane ground to environment
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

# Run the Simulation
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
while True:
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
