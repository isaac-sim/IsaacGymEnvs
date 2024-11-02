import time

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.drone_racing.env import (
    EnvCreatorParams,
    EnvCreator,
)
from isaacgymenvs.tasks.drone_racing.managers import (
    DroneManager,
    DroneManagerParams,
    RandDroneOptions,
)
from isaacgymenvs.tasks.drone_racing.managers import (
    ObstacleManager,
    RandObstacleOptions,
)
from isaacgymenvs.tasks.drone_racing.waypoint import (
    WaypointGenerator,
    WaypointGeneratorParams,
    RandWaypointOptions,
)

print("import torch")
import torch

if __name__ == "__main__":
    # settings
    draw_orbit_min = False
    draw_orbit_mean = False
    draw_orbit_max = False
    draw_wall_region = False
    run_physics_sim = True
    torch.manual_seed(42)

    # create sim and gym
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # create envs
    env_creator_params = EnvCreatorParams()
    env_creator_params.disable_tqdm = False
    env_creator_params.num_envs = 1
    print("Initializing environment creator...")
    env_creator = EnvCreator(gym, sim, env_creator_params)
    print("Creating envs and actors...")
    env_creator.create([0.0, 0.0, env_creator_params.env_size / 2])

    # all environments are set up, prepare sim
    gym.prepare_sim(sim)
    actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)

    # init waypoint generator
    wp_generator_params = WaypointGeneratorParams()
    wp_generator_params.num_envs = env_creator_params.num_envs
    wp_generator_params.num_waypoints = 4
    wp_generator_params.num_gate_x_lens = len(env_creator_params.gate_bar_len_x)
    wp_generator_params.num_gate_weights = len(env_creator_params.gate_bar_len_z)
    wp_generator_params.gate_weight_max = max(env_creator_params.gate_bar_len_z)
    wp_generator_params.fixed_waypoint_id = 1
    wp_generator_params.fixed_waypoint_position = [
        0.0,
        0.0,
        env_creator_params.env_size / 2,
    ]
    wp_generator = WaypointGenerator(wp_generator_params)

    # init actor manager
    obstacle_manager = ObstacleManager(env_creator)
    rand_obs_opts = RandObstacleOptions()
    rand_obs_opts.wall_density = 0
    rand_obs_opts.tree_density = 0
    drone_manager_params = DroneManagerParams()
    drone_manager_params.num_envs = env_creator_params.num_envs
    drone_manager = DroneManager(drone_manager_params)

    # viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "randomize")
    print("Use R key to randomize envs.")

    # update simulation
    while not gym.query_viewer_has_closed(viewer):

        # check reset key
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "randomize" and evt.value > 0:

                # generate random waypoints
                t0 = time.time()
                rand_wp_options = RandWaypointOptions()
                rand_wp_options.init_roll_max = 0
                rand_wp_options.init_pitch_max = 0
                rand_wp_options.init_yaw_max = 0
                rand_wp_options.psi_max = 0
                rand_wp_options.theta_max = 0
                # rand_wp_options.alpha_max = 0
                rand_wp_options.gamma_max = 0
                rand_wp_options.r_min = 10
                rand_wp_options.r_max = 10
                rand_wp_options.force_gate_flag = 1
                wp_data = wp_generator.compute(rand_wp_options)

                # update obstacles
                actor_pose, actor_id = obstacle_manager.compute(
                    waypoint_data=wp_data, rand_obs_opts=rand_obs_opts
                )
                actor_root_state[actor_id, :7] = actor_pose[actor_id].to("cuda")

                # update drone
                drone_manager.set_waypoint(wp_data)
                drone_state, act, next_id = drone_manager.compute(RandDroneOptions())
                actor_root_state[env_creator.drone_actor_id.flatten()] = drone_state

                # submit
                gym.set_actor_root_state_tensor(
                    sim, gymtorch.unwrap_tensor(actor_root_state)
                )

                # draw debug geometries
                t1 = time.time()
                gym.clear_lines(viewer)
                wp_data.visualize(gym, env_creator.envs, viewer, 1)
                orbit_vis_data, wall_vis_data = obstacle_manager.get_vis_data()
                orbit_vis_data.visualize(
                    gym,
                    env_creator.envs,
                    viewer,
                    draw_min=draw_orbit_min,
                    draw_mean=draw_orbit_mean,
                    draw_max=draw_orbit_max,
                )
                if draw_wall_region:
                    wall_vis_data.visualize(gym, env_creator.envs, viewer)
                t2 = time.time()

                # print time info
                print("---")
                print("envs rand:", int((t1 - t0) * 1000), "ms")
                print("debug vis:", int((t2 - t1) * 1000), "ms")

        # gym.simulate(sim)
        if run_physics_sim:
            gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_sim(sim)
