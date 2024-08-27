import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(project_base_dir)

import numpy as np
import os
import argparse
import cv2
import time
import shutil

from ball_rolling_exp.ball_rolling_tactile_world import BallRollingTactileWorld, parse_args
from isaacgymenvs.tacsl_sensors.shear_tactile_viz_utils import visualize_tactile_shear_image

import torch

if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)

    # parse arguments
    args = parse_args()

    args.headless = not args.render
    
    # set torch device
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'

    tactile_world = BallRollingTactileWorld(args)
    tactile_world.setup_viewer_camera()

    #### Initialize shear force computation
    tactile_world.step_physics()
    num_divs = [args.num_tactile_rows, args.num_tactile_cols]
    tactile_world.initialize_penalty_based_tactile(num_divs = num_divs, sdf_tool=args.sdf_tool)

    # generate control sequence, the pad is force/torque controlled with a 3-DoF translational joint
    action_array = [np.array([0., 0., 0.2]), 
                    np.array([0.007, 0., 0.2]), np.array([-0.007, 0., 0.2]), 
                    np.array([-0.007, 0., 0.2]), np.array([0.007, 0., 0.2]), 
                    np.array([0., 0.007, 0.2]), np.array([0., -0.007, 0.2]),
                    np.array([0., -0.007, 0.2]), np.array([0., 0.007, 0.2])]
    # steps_array = [0, 100, 140, 170, 200, 230, 270, 300, 330, 360] # mesh
    steps_array = [0, 100, 130, 160, 190, 220, 250, 280, 310, 340] # sphere


    
    if args.render:
        tactile_world.freeze_sim_and_render()
    
    if args.record:
        tactile_img_folder = 'tactile_record'
        if os.path.exists(tactile_img_folder):
            shutil.rmtree(tactile_img_folder)
            
        os.makedirs(tactile_img_folder, exist_ok = True)
        os.makedirs(os.path.join(tactile_img_folder, 'force_map'), exist_ok = True)
        
    actions = []
    for i in range(len(steps_array) - 1):
        for j in range(steps_array[i], steps_array[i + 1]):
            actions.append(torch.tensor(action_array[i], dtype=torch.float, device=device).unsqueeze(0).repeat(args.num_envs, 1))

    tactile_cnt = 0
    t_start = time.time()
    t_sum_tactile = 0
    t_sum_physics = 0
    for step in range(len(actions)):
        t_start_physics = time.time()
        tactile_world.apply_action_dof_torque(actions[step])
        tactile_world.step_physics()
        tactile_world.refresh_tensors()
        t_end_tactile = time.time()
        t_sum_physics += t_end_tactile - t_start_physics

        if not args.disable_tactile:
            if step % args.tactile_frequency == 0:
                t_start_tactile = time.time()
                penetration_depth, tactile_normal_force, tactile_shear_force = tactile_world.get_penalty_based_tactile_forces()
                t_end_tactile = time.time()
                t_sum_tactile += t_end_tactile - t_start_tactile
                tactile_cnt += 1
                # visualize tactile forces

                if args.render:
                    penetration_depth = penetration_depth[0].cpu().numpy()
                    # print('max penetration = ', penetration_depth.max())
                    
                    nrows = num_divs[0]
                    ncols = num_divs[1]
                    
                    tactile_normal_force = tactile_normal_force[0].view((nrows, ncols))
                    tactile_shear_force = tactile_shear_force[0].view((nrows, ncols, 2))
                    
                    num_nonzero_normal_forces = (tactile_normal_force != 0).sum()
                    num_nonzero_shear_forces = (tactile_shear_force.sum(dim = -1) != 0).sum()
                    
                    # print('non zeros normal forces = ', num_nonzero_normal_forces, ', non zeros shear forces = ', num_nonzero_shear_forces)
                    tactile_image = visualize_tactile_shear_image(
                                        tactile_normal_force.detach().cpu().numpy(), 
                                        tactile_shear_force.detach().cpu().numpy(),
                                        normal_force_threshold = 0.0008,
                                        shear_force_threshold = 0.0004)
                    
                    if args.record:
                        cv2.imwrite(os.path.join(os.path.join(tactile_img_folder, 'force_map'), '{}.png'.format(tactile_cnt)), tactile_image * 255)
                    cv2.imshow('tactile_force', tactile_image)
                    cv2.waitKey(1)
            
        # update viewer
        if args.render:
            tactile_world.render_viewer()
    
    t_end = time.time()
    
    FPS = len(actions) * args.num_envs / (t_end - t_start)
    FPS_physics = len(actions) * args.num_envs / t_sum_physics
    FPS_tactile = len(actions) * args.num_envs / t_sum_tactile
    print(f'\nNum envs = {args.num_envs}',
          f'\ntime elapsed = {t_end - t_start}, time_physics = {t_sum_physics}, time_tactile = {t_sum_tactile}',
          f'\nTotal FPS = {FPS}, Physics FPS = {FPS_physics}, Tactile FPS = {FPS_tactile} \n')
    
    if args.render:
        tactile_world.freeze_sim_and_render()
    
    # 200x200
    # Num_Envs        FPS
    #        1        142
    #        8       1227
    #       64       7715
    #      512      22933
    
    # 20x20
    # Num_Envs        FPS
    #        1        161
    #        8       1582
    #       64       9277
    #      512      71381
    #     4096     425391
    #    32768    1002476