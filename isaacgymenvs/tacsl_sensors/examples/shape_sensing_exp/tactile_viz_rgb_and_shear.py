'''
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visuo-Tactile Sensing
----------------
Use tactile sensing (RGB and shear force field) as mesh interacts with tactile-enabled finger.
'''

from shape_sensing_world_ige import TactileWorld, parse_args

import cv2
import numpy as np
import os
import time
import torch
from isaacgymenvs.tacsl_sensors.shear_tactile_viz_utils import visualize_penetration_depth, visualize_tactile_shear_image


if __name__ == '__main__':
    # set random seed
    np.random.seed(42)

    torch.set_printoptions(precision=4, sci_mode=False)

    # parse arguments
    args = parse_args()

    args.headless = not args.render

    # set torch device
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'

    # Create simulation
    tactile_world = TactileWorld(args)
    tactile_world.setup_viewer_camera()

    # move the indenters directly on top of points on the elastomer.
    elastomer_node_pos_list_per_env, elastomer_node_quat_list_per_env = tactile_world.generate_grid_on_elastomer(
        num_divs=1)
    indenter_start_pos_list_per_env, indenter_start_quat_list_per_env = tactile_world.compute_indenter_start_poses_for_elastomer_nodes(
        elastomer_node_pos_list_per_env, elastomer_node_quat_list_per_env, args.distance_to_sensor)

    # set starting pose of indenter
    num_grid_points = indenter_start_pos_list_per_env.shape[1]
    for node_id in range(num_grid_points):
        tactile_world.reset_indenter_pose(indenter_start_pos_list_per_env[:, node_id],
                                          indenter_start_quat_list_per_env[:, node_id])
        tactile_world.step_physics()
        tactile_world.refresh_tensors()

    # Teleport indenter to starting pose
    if not args.floating_indenter:
        starting_dof = tactile_world.dof_pos.detach().clone()
        starting_dof[:, 2] = -args.distance_to_sensor   # set starting distance from the elastomer, when 0, indenter is at -args.distance_to_sensor
        tactile_world.reset_indenter_dof(starting_dof)
        tactile_world.step_physics()
        tactile_world.refresh_tensors()

    if args.render:
        tactile_world.render_viewer()
        print('\n\n Ctrl + C to continue...')
        tactile_world.freeze_sim_and_render()

    # Interactively move indenter until contact
    if not args.floating_indenter:
        ctrl_action = torch.zeros_like(tactile_world.dof_pos)
        ctrl_action_increment_pos = 1e-2
        ctrl_action_increment_rot = 1e-2
        ctrl_action_pos_max = torch.tensor([1e-3, 1e-3, 1e-0], device=tactile_world.device)
        ctrl_action_rot_max = 0.2

    # programatically move indenter around
    translation_force = 3.0/2
    rotation_torque = 0.05/2
    action_sequence = [
        [0., 0., -translation_force, 0., 0., 0.],  # move down
        [0.0, 0., -translation_force, 0., 0., rotation_torque],  # rotate
    ]

    rotation_torque = 0.001
    sim_params = tactile_world.gym.get_sim_params(tactile_world.sim)
    sim_dt = sim_params.dt
    num_sim_steps_per_action = round(args.action_duration / sim_dt)

    action_sequence = np.array(action_sequence)

    if args.record:
        tactile_img_folder = 'tactile_record'
        os.makedirs(tactile_img_folder, exist_ok = True)
        tactile_rgb_dir = os.path.join(tactile_img_folder, 'tactile_rgb')
        tactile_ff_dir = os.path.join(tactile_img_folder, 'force_field')
        os.makedirs(tactile_rgb_dir, exist_ok = True)
        os.makedirs(tactile_ff_dir, exist_ok = True)

    time_costs = {'physics': 0, 'rgb_tactile': 0, 'numerical_tactile': 0}
    tactile_cnt = 0
    t_start = time.time()
    for a_id in range(len(action_sequence)):
        action = action_sequence[a_id]
        for step in range(num_sim_steps_per_action):
            t_start_physics = time.time()
            # refresh tensors
            tactile_world.refresh_tensors()

            # step the physics
            tactile_world.step_physics()
            if not args.floating_indenter:
                ctrl_action[:, :6] = torch.tensor(action, device=tactile_world.device)
                # set dof torques
                tactile_world.apply_action_dof_torque(ctrl_action)
            time_costs['physics'] += time.time() - t_start_physics

            tactile_cnt += 1
            # Get Tactile RGB
            t_start_tactile_rgb = time.time()
            if args.use_tactile_rgb:
                tactile_world.refresh_tensors()
                images = tactile_world.get_camera_image_tensors_dict()
            time_costs['rgb_tactile'] += time.time() - t_start_tactile_rgb

            if (args.render or args.record) and args.use_tactile_rgb:
                image_tiled_list = [images[k][0].cpu().numpy() for k in images.keys() if 'taxim' in k]

                if image_tiled_list:
                    image_tiled_all = np.concatenate(image_tiled_list, axis=0)
                    image_tiled_all = image_tiled_all[:, ::-1]    # mirror tactile image along y

                    if args.record:
                        cv2.imwrite(os.path.join(tactile_rgb_dir, '{}.png'.format(tactile_cnt)), image_tiled_all[..., ::-1])
                    if args.render:
                        if len(image_tiled_all.shape) > 2:
                            cv2.imshow('Gym Images', image_tiled_all[..., ::-1])
                        else:
                            cv2.imshow('Gym Images Depth', cv2.normalize(image_tiled_all, None, 0, 1, cv2.NORM_MINMAX))
                        cv2.waitKey(1)

            # Get Tactile Force Field
            t_start_tactile_ff = time.time()
            if args.use_tactile_ff:
                tactile_force_field_dict = tactile_world.get_tactile_shear_force_fields()
            time_costs['numerical_tactile'] += time.time() - t_start_tactile_ff

            if (args.render or args.record) and args.use_tactile_ff:
                tactile_ff_key = list(tactile_force_field_dict.keys())[0]
                penetration_depth, tactile_normal_force, tactile_shear_force = tactile_force_field_dict[tactile_ff_key]
                nrows, ncols = args.num_tactile_rows, args.num_tactile_cols
                penetration_depth = penetration_depth.view((args.num_envs, nrows, ncols))
                penetration_depth_img_upsampled = visualize_penetration_depth(
                    penetration_depth[0].detach().cpu().numpy(), resolution=5, depth_multiplier=300.)

                # visualize tactile forces
                tactile_normal_force = tactile_normal_force.view((args.num_envs, nrows, ncols))
                tactile_shear_force = tactile_shear_force.view((args.num_envs, nrows, ncols, 2))

                tactile_image = visualize_tactile_shear_image(
                    tactile_normal_force[0].detach().cpu().numpy(),
                    tactile_shear_force[0].detach().cpu().numpy())

            if args.record and args.use_tactile_ff:
                cv2.imwrite(os.path.join(tactile_ff_dir, '{}.png'.format(tactile_cnt)), tactile_image * 255)

            if args.render and args.use_tactile_ff:
                cv2.imshow('penetration_depth', penetration_depth_img_upsampled)
                cv2.waitKey(1)
                cv2.imshow('Tactile Force Field', tactile_image)
                cv2.waitKey(1)

            # update viewer
            if args.render:
                tactile_world.render_viewer()

    t_end = time.time()
    num_frames = len(action_sequence) * num_sim_steps_per_action * args.num_envs
    FPS = num_frames / (t_end - t_start)
    FPS_physics = num_frames / time_costs['physics']
    FPS_tactile_img = num_frames / time_costs['rgb_tactile']
    FPS_tactile_ff = num_frames / time_costs['numerical_tactile']
    print(f'\nNum envs = {args.num_envs}',
          f'\ntime elapsed = {(t_end - t_start)/num_frames}, time_physics = {time_costs["physics"]/num_frames}, '
          f'time_tactile_rgb = { time_costs["rgb_tactile"]/num_frames}, time_tactile_ff = {time_costs["numerical_tactile"]/num_frames}',
          f'\nTotal FPS = {FPS}, Physics FPS = {FPS_physics}, '
          f'Tactile Img FPS = {FPS_tactile_img}, Tactile FF FPS = {FPS_tactile_ff} \n')

    tactile_world.clean_up()