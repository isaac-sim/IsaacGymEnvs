import argparse
import pathlib

from typing import Dict, Tuple
import numpy as np 
import json 
import yaml
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert a1_expert_raw to desired format")
    parser.add_argument("-i", "--input-dir", type=str, default='data/motions/quadruped/mania_pos_rew')
    parser.add_argument("-o", "--output-dir", type=str, default='data/motions/quadruped/mania_pos')
    parser.add_argument("-n", "--dataset-name", type=str, default="dataset")
    parser.add_argument("-s", "--start-time-frac", type=float, help="Start time as a fraction. E.g. 0.2 = start from 20% of the way in", default = 0.0)
    parser.add_argument("-e", "--end-time-frac", type=float, help="End time as a fraction. E.g. 0.8 = end at 80% of the way in", default = 1.0)
    args = parser.parse_args()
    return args

def read_npz(filepath: str) -> np.ndarray:
    return np.load(filepath)['arr_0']

def read_csv(filepath: str, delimiter=',') -> np.ndarray:
    return np.genfromtxt(filepath, delimiter=delimiter)

def reorder_dofs(dof_data: np.ndarray) -> np.ndarray:
    """
    Input:  FL_hip'  FL_thigh' FL_calf FR_hip FR_thigh FR_calf RL_hip RL_thigh RL_calf RR_hip RR_thigh RR_calf
    Output: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    Input: (n_dof, ...)
    Output: (n_dof, ...)
    """
    fr_data = dof_data[3:6, ...]
    fl_data = dof_data[:3, ...]
    rr_data = dof_data[9:12, ...]
    rl_data = dof_data[6:9, ...]
    return np.concatenate([fl_data, fr_data, rl_data, rr_data], axis=0)

def write_motion_data(filepath: str, frames: np.ndarray, dt: float, loop_mode: str = 'Clamp', enable_cycle_offset_position: bool = True, enable_cycle_offset_rotation: bool = False):


    motion_data = {}
    motion_data["LoopMode"] = loop_mode
    motion_data["Frames"] = frames.tolist()
    motion_data["FrameDuration"] = dt
    motion_data["EnableCycleOffsetPosition"] = enable_cycle_offset_position
    motion_data["EnableCycleOffsetRotation"] = enable_cycle_offset_rotation
    with open(filepath, 'w') as file:
        json.dump(motion_data, file)

def write_dataset(motion_fps: str, dataset_fp: str):
    dataset = []
    # Uniformly weight all files
    weight = round(1 / len(motion_fps), 5) 
    for fp in motion_fps: 
        dataset.append({
            'file': fp.name,
            'weight': weight
        })
    with open(dataset_fp, 'w') as file:
        yaml.dump({'motions': dataset}, file)

if __name__ == "__main__":
    args = parse_args()

    # Set up directories

    path_folder = os.path.dirname(os.getcwd())
    file_dir = pathlib.Path(args.input_dir)
    folder_dir = pathlib.Path(path_folder)

    input_dir = folder_dir/file_dir



    output_filepaths = []




    for i in range(2):
        i=i+6

        nums = [3,4,5]
        if (i in nums): continue # No 5 for some reason
        filepaths = [
            f'base_position{i}.npz',
            f'base_orientation{i}.npz',
            f'joint_angles{i}.npz'
        ]
        filepaths = [input_dir / fp for fp in filepaths]

        base_position = np.load(filepaths[0])['base_position']
        base_orientation = np.load(filepaths[1])['base_orientation']
        joint_angles = np.load(filepaths[2])['joint_angles']
        joint_angles = reorder_dofs(joint_angles.T).T


        base_position1 =  np.repeat(base_position, 2)
        base_orientation1 = np.repeat(base_orientation, 2)
        joint_angles1 = np.repeat(joint_angles, 2)

        # # Let's assume you have the following numpy arrays:
        # timestamps = np.arange(0, 10, 0.001)  # original timestamps with 0.01s interval
        # new_timestamps = np.arange(0, 10, 0.002)
        #
        # print(timestamps.shape)
        # print(base_position.shape)
        #
        bp = np.empty((base_position.shape[0] *2, base_position.shape[1]))
        bo = np.empty((base_position.shape[0] *2, base_orientation.shape[1]))
        ja = np.empty((base_position.shape[0] *2, joint_angles.shape[1]))

        # Interpolate each feature separately
        for j in range(base_position.shape[1]):
            bp[:,j] = np.repeat(base_position[:,j], 2)
        for l in range(base_orientation.shape[1]):
            bo[:,l] = np.repeat(base_orientation[:,l], 2)
        for m in range(joint_angles.shape[1]):
            ja[:,m] = np.repeat(joint_angles[:,m], 2)




        frame_data = np.concatenate([bp, bo, ja], axis=-1)
        # frame_data = np.concatenate([base_position, base_orientation, joint_angles], axis=-1)

        # Write motion data

        output_file= pathlib.Path(args.output_dir)
        output_dir = folder_dir/output_file


        output_dir.mkdir(parents=True, exist_ok=True)
        output_filepath = output_dir / f'motion{i}.txt'
        output_filepaths.append(output_filepath)
        write_motion_data(output_filepath, frame_data, dt=0.001)

        i = i+2

    write_dataset(output_filepaths, output_dir / f'{args.dataset_name}.yaml')

    # Plot motion data
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3)
    # ax[0].plot(frame_data[:, 0:3])
    # ax[1].plot(frame_data[:, 3:7])
    # ax[2].plot(frame_data[:, 7:19])
    # fig.show()
    input("Press Enter to continue...")

