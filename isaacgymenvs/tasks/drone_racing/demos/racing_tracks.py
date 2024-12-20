from typing import List, Tuple

from isaacgym import gymapi
from isaacgym.gymapi import Asset
from isaacgymenvs.tasks.drone_racing.assets import (
    TrackMultiStoryOptions,
    TrackRmuaOptions,
    TrackSplitsOptions,
    TrackWallsOptions,
    TrackGeomKebabOptions,
    TrackPlanarCircleOptions,
    TrackWavyEightOptions,
    TrackTurnsOptions,
    TrackSimpleStickOptions,
    create_track_multistory,
    create_track_rmua,
    create_track_splits,
    create_track_walls,
    create_track_geom_kebab,
    create_track_planar_circle,
    create_track_wavy_eight,
    create_track_turns,
    create_track_simple_stick,
)
from isaacgymenvs.tasks.drone_racing.waypoint import (
    Waypoint,
    WaypointData,
)

print("Importing torch...")
import torch  # noqa


def define_track_assets() -> Tuple[List[Asset], List[str], List[List[Waypoint]]]:
    track_assets = []
    track_names = []
    track_wp_lists = []

    # multistory without debug view
    multistory_options = TrackMultiStoryOptions()
    multistory_asset, multistory_wp = create_track_multistory(
        gym, sim, multistory_options
    )
    track_assets.append(multistory_asset)
    track_names.append("multistory")
    track_wp_lists.append(multistory_wp)

    # multistory with debug view
    multistory_options.track_options.enable_debug_visualization = True
    multistory_asset, multistory_wp = create_track_multistory(
        gym, sim, multistory_options
    )
    track_assets.append(multistory_asset)
    track_names.append("multistory_debug")
    track_wp_lists.append(multistory_wp)

    # rmua simple with debug view
    rmua_options = TrackRmuaOptions()
    rmua_options.enable_waypoint_randomization = False
    rmua_options.enable_additional_obstacles = False
    rmua_options.track_options.enable_debug_visualization = True
    rmua_asset, rmua_wp = create_track_rmua(gym, sim, rmua_options)
    track_assets.append(rmua_asset)
    track_names.append("rmua_simple")
    track_wp_lists.append(rmua_wp)

    # rmua random waypoint with debug view
    rmua_options.enable_waypoint_randomization = True
    rmua_options.enable_additional_obstacles = False
    rmua_options.track_options.enable_debug_visualization = True
    rmua_asset, rmua_wp = create_track_rmua(gym, sim, rmua_options)
    track_assets.append(rmua_asset)
    track_names.append("rmua_waypoint")
    track_wp_lists.append(rmua_wp)

    # rmua add more obstacles
    rmua_options.enable_waypoint_randomization = True
    rmua_options.enable_additional_obstacles = True
    rmua_options.track_options.enable_debug_visualization = True
    rmua_asset, rmua_wp = create_track_rmua(gym, sim, rmua_options)
    track_assets.append(rmua_asset)
    track_names.append("rmua_obstacles_1")
    track_wp_lists.append(rmua_wp)

    # rmua add more obstacles
    rmua_options.enable_waypoint_randomization = True
    rmua_options.enable_additional_obstacles = True
    rmua_options.track_options.enable_debug_visualization = False
    rmua_asset, rmua_wp = create_track_rmua(gym, sim, rmua_options)
    track_assets.append(rmua_asset)
    track_names.append("rmua_obstacles_2")
    track_wp_lists.append(rmua_wp)

    # split-s
    splits_options = TrackSplitsOptions()
    splits_asset, splits_wp = create_track_splits(gym, sim, splits_options)
    track_assets.append(splits_asset)
    track_names.append("splits_debug")
    track_wp_lists.append(splits_wp)

    # walls
    walls_options = TrackWallsOptions()
    walls_asset, walls_wp = create_track_walls(gym, sim, walls_options)
    track_assets.append(walls_asset)
    track_names.append("walls_debug")
    track_wp_lists.append(walls_wp)

    # obstacle kebab
    geom_kebab_options = TrackGeomKebabOptions()
    geom_kebab_asset, geom_kebab_wp = create_track_geom_kebab(
        gym, sim, geom_kebab_options
    )
    track_assets.append(geom_kebab_asset)
    track_names.append("geom_kebab")
    track_wp_lists.append(geom_kebab_wp)

    # planar circle
    planar_circle_options = TrackPlanarCircleOptions()
    planar_circle_asset, planar_circle_wp = create_track_planar_circle(
        gym, sim, planar_circle_options
    )
    track_assets.append(planar_circle_asset)
    track_names.append("planar_circle")
    track_wp_lists.append(planar_circle_wp)

    # wavy eight
    wavy_eight_options = TrackWavyEightOptions()
    wavy_eight_asset, wavy_eight_wp = create_track_wavy_eight(
        gym, sim, wavy_eight_options
    )
    track_assets.append(wavy_eight_asset)
    track_names.append("wavy_eight")
    track_wp_lists.append(wavy_eight_wp)

    # turns
    turns_options = TrackTurnsOptions()
    turns_asset, turns_wp = create_track_turns(gym, sim, turns_options)
    track_assets.append(turns_asset)
    track_names.append("turns")
    track_wp_lists.append(turns_wp)

    # simple stick
    simple_stick_options = TrackSimpleStickOptions()
    simple_stick_asset, simple_stick_wp = create_track_simple_stick(
        gym, sim, simple_stick_options
    )
    track_assets.append(simple_stick_asset)
    track_names.append("simple_stick")
    track_wp_lists.append(simple_stick_wp)

    # TODO: more tracks can be added here

    return track_assets, track_names, track_wp_lists


if __name__ == "__main__":
    # create sim and gym
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # spawn track assets into envs
    envs = []
    assets, names, wp_lists = define_track_assets()
    assert len(assets) == len(names)
    for i in range(len(assets)):
        env = gym.create_env(
            sim,
            gymapi.Vec3(-20, -20, 0),
            gymapi.Vec3(20, 20, 40),
            4,
        )
        gym.create_actor(env, assets[i], gymapi.Transform(), names[i], i, 1)
        envs.append(env)

    # viewer and gpu pipeline
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(
        viewer,
        None,
        gymapi.Vec3(-20, -20, 30),
        gymapi.Vec3(20, 20, 10),
    )
    gym.prepare_sim(sim)

    # draw waypoint data on those w/o debug views
    torch.set_printoptions(linewidth=130, sci_mode=False, precision=2)
    for env_id in [0, 5, 6, 7, 8, 9, 10, 11, 12]:
        wp_data = WaypointData.from_waypoint_list(1, wp_lists[env_id])
        wp_data.visualize(gym, [envs[env_id]], viewer, 1)
        print("---")
        print("env  :", env_id)
        print("r    :", wp_data.r)
        print("psi  :", torch.rad2deg(wp_data.psi))
        print("theta:", torch.rad2deg(wp_data.theta))
        print("gamma:", torch.rad2deg(wp_data.gamma))

    # update simulation
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_sim(sim)
