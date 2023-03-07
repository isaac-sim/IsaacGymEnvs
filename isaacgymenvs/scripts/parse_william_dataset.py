import argparse
import pathlib

from typing import Dict, Tuple
import numpy as np 
import json 
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert a1_expert_raw to desired format")
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("-n", "--dataset-name", type=str, default="dataset")
    parser.add_argument("-s", "--start-time-frac", type=float, help="Start time as a fraction. E.g. 0.2 = start from 20% of the way in", default = 0.0)
    parser.add_argument("-e", "--end-time-frac", type=float, help="End time as a fraction. E.g. 0.8 = end at 80% of the way in", default = 1.0)
    args = parser.parse_args()
    return args

def read_csv(filepath: str, delimiter=',') -> np.ndarray:
    return np.genfromtxt(filepath, delimiter=delimiter)

def reorder_dofs(dof_data: np.ndarray) -> np.ndarray:
    """
    Input: FR_HAA_Pos,FR_HFE_Pos,FR_KFE_Pos,FL_HAA_Pos,FL_HFE_Pos,FL_KFE_Pos,RR_HAA_Pos,RR_HRE_Pos,RR_KRE_Pos,RL_HAA_Pos,RL_HRE_Pos,RL_KRE_Pos
    Output: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    Input: (n_dof, ...)
    Output: (n_dof, ...)
    """
    fr_data = dof_data[:3, ...]
    fl_data = dof_data[3:6, ...]
    rr_data = dof_data[6:9, ...]
    rl_data = dof_data[9:12, ...]
    return np.concatenate([fl_data, fr_data, rl_data, rr_data], axis=0)

def parse_mocap_data(filepath: str) -> Tuple[np.ndarray, float]:
    """ 
    Input: Filepath to raw data 
    Output: (T, 19) array, each row is (body_pos, body_orn, dof_pos)
    """
    arr = read_csv(filepath)
    time = arr[:,0]
    dt = time[1] - time[0] # assume constant timestep
    body_pos = arr[:,1:4] # x, y, z
    body_orn = arr[:,4:8] # x, y, z, w
    dof_pos = reorder_dofs(arr[:,8:20].T).T # (FL, FR, RL, RR) sequence of (hip, thigh, calf) 
    frame_data = np.concatenate([body_pos, body_orn, dof_pos], axis=-1)
    return frame_data, dt

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
    weight = round(1 / len(filepaths), 5) 
    for fp in motion_fps: 
        dataset.append({
            'file': fp.name + '.txt',
            'weight': weight
        })
    with open(dataset_fp, 'w') as file:
        yaml.dump({'motions': dataset}, file)

if __name__ == "__main__":
    args = parse_args()

    # Set up directories
    input_dir = pathlib.Path(args.input_dir)
    filepaths = []
    for p in input_dir.rglob("*"):
        if p.suffix == '.csv':
            filepaths.append(p.absolute())
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse mo-cap data
    for fp in filepaths:
        frame_data, dt = parse_mocap_data(str(fp))
        n_timesteps = frame_data.shape[0]
        start_time = int(n_timesteps * args.start_time_frac)
        end_time = int(n_timesteps * args.end_time_frac)
        output_fp = output_dir / (fp.name + '.txt')
        write_motion_data(
            str(output_fp), 
            frame_data[start_time: end_time], 
            dt, loop_mode='Clamp'
        )

    # Write combined dataset
    dataset_fp = output_dir / (args.dataset_name + '.yaml')
    write_dataset(filepaths, dataset_fp)
        