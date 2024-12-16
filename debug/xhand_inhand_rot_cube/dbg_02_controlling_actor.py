# import libraries
from isaacgym import gymapi, gymtorch
import torch

# initialize isaac gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# create the environments
envs = []
num_envs = 4
for i in range(num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), int(num_envs ** 0.5))
    envs.append(env)

    # load assets
    asset_root = "./assets" # "path/to/assets"
    robot_asset_file = "urdf/xhand/xhand_right.urdf"# "urdf/robot.urdf"
    robot_asset_options = gymapi.AssetOptions()
    robot_asset_options.fix_base_link = True
    robot_asset_options.disable_gravity = True
    robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, robot_asset_options)

    # load actors
    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0, 0, 0.7)  # Position (x, y, z)
    actor_handle=gym.create_actor(env, robot_asset, robot_pose, "robot", i, 0)
    
    # define actor dof properties
    dof_props = gym.get_asset_dof_properties(robot_asset)
    # dof_props['effort'].fill = 100.0
    dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    dof_props['stiffness'].fill(3.0)
    dof_props['damping'].fill(0.1)
    dof_props['friction'].fill(0.01)
    dof_props['armature'].fill(0.001)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)
        
    # get actor dof properties
    num_dofs = gym.get_actor_dof_count(env, actor_handle)
    dof_lower_limits = dof_props['lower']
    dof_upper_limits = dof_props['upper']
    
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

# Flag to toggle between applying actions per environment or all environments
APPLY_PER_ENV = True  # Set to False to apply actions to all environments at once

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# adjust the camera position to focus on the robot
# Get robot's position
robot_position = gymapi.Vec3(0, 0, 0)  # Replace with your robot's position
focus_distance = 2.0  # Distance from the robot to the camera

# Set camera position and look-at
camera_position = gymapi.Vec3(
    robot_position.x - focus_distance, 
    robot_position.y, 
    robot_position.z + focus_distance
)
gym.viewer_camera_look_at(viewer, None, camera_position, robot_position)

while not gym.query_viewer_has_closed(viewer):
    # step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # Refresh root state tensors
    # gym.refresh_actor_root_state_tensor(sim)
    

    # Apply actions per environment
    for env in envs:
        for i in range(gym.get_actor_count(env)):
            # Get DOF states
            dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

            # Separate joint positions and velocities
            joint_positions = [state["pos"] for state in dof_states]
            print(f'Joint positions: {joint_positions}')
            actor_handle = gym.get_actor_handle(env, i)
            num_dofs = gym.get_actor_dof_count(env, actor_handle)
            action_tensor = torch.tensor(joint_positions) + 0.05
            print(f'Action tensor: {action_tensor}')
            # action_tensor = torch.zeros((num_dofs,), dtype=torch.float32, device="cuda:0")
            # action_tensor[0] = 0.1  # Example: Setting position of the first joint
            gym.set_actor_dof_position_targets(env, actor_handle, action_tensor.cpu().numpy())

    # visualize the simulation
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # wait for the viewer to process events
    gym.sync_frame_time(sim)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)