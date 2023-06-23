import isaacgym
import numpy as np
import os
import argparse
import pathlib
import torch
from isaacgym import gymutil, gymapi, gymtorch
from isaacgymenvs.utilities.quadruped_motion_data import MotionLib

def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')
    parser.add_argument('-i', '--input-filepath', type=str, default='data/motions/quadruped/mania_pos/motion7.txt')
    # parser.add_argument('-i', '--input-filepath', type=str, default='data/motions/quadruped/mania_pos/dataset.yaml')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args

def set_env_state(
        gym, sim, 
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):

    root_state = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
    dof_state = torch.stack([dof_pos, dof_vel], dim=-1)
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    return

def init_camera(gym, sim, viewer):
    gym.refresh_actor_root_state_tensor(sim)
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(_root_tensor)

    _cam_prev_char_pos = root_states[0, 0:3].cpu().numpy()
    
    cam_pos = gymapi.Vec3(_cam_prev_char_pos[0], 
                            _cam_prev_char_pos[1] - 3.0, 
                            1.0)
    cam_target = gymapi.Vec3(_cam_prev_char_pos[0],
                                _cam_prev_char_pos[1],
                                1.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    return _cam_prev_char_pos

def update_camera(gym, sim, viewer, root_pos, prev_char_root_pos):
    char_root_pos = root_pos.cpu().numpy()
    
    cam_trans = gym.get_viewer_camera_transform(viewer, None)
    cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
    cam_delta = cam_pos - prev_char_root_pos

    new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
    new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                char_root_pos[1] + cam_delta[1], 
                                cam_pos[2])

    gym.viewer_camera_look_at(viewer, None, new_cam_pos, new_cam_target)

    prev_char_root_pos[:] = char_root_pos
    return prev_char_root_pos

if __name__ == "__main__":
    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = parse_arguments()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_FLEX:
        print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
        args.physics_engine = gymapi.SIM_PHYSX
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # load A1 asset
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
    asset_file = "urdf/a1.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # set up the env grid
    num_envs = 1
    num_per_row = 80
    env_spacing = 0.56
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    pose.p.z = 1.
    pose.p.x = 40.


    envs = []
    # set random seed
    np.random.seed(17)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # generate random bright color
        c = 0.5 + 0.5 * np.random.random(3)
        color = gymapi.Vec3(c[0], c[1], c[2])

        ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
        gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # load motion library
    project_dir = pathlib.Path(__file__).absolute().parent.parent
    motion_filepath = str(project_dir / args.input_filepath)
    motion_lib = MotionLib(motion_filepath, device=torch.device('cpu'))

    # create a local copy of initial state, which we can send back for reset
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    prev_char_pos = init_camera(gym, sim, viewer)

    # subscribe to spacebar event for reset
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")


    time = 0
    while not gym.query_viewer_has_closed(viewer):

        # Get input actions from the viewer and handle them appropriately
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "reset" and evt.value > 0:
                gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        if time * sim_params.dt > motion_lib.get_motion_length([0]):
            time = 0

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = motion_lib.get_motion_state([0], [time * sim_params.dt])
        set_env_state(gym, sim, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
        prev_char_pos = update_camera(gym, sim, viewer, root_pos[0], prev_char_pos)


        print(root_ang_vel)



        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        time += 1

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)