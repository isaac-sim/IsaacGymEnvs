import argparse
import pathlib

from scipy.spatial.transform import Rotation
from typing import Dict, Tuple
import numpy as np 
import json 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert a1_expert_raw to desired format")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("-s", "--start-time-frac", type=float, help="Start time as a fraction. E.g. 0.2 = start from 20% of the way in", default = 0.0)
    parser.add_argument("-e", "--end-time-frac", type=float, help="End time as a fraction. E.g. 0.8 = end at 80% of the way in", default = 1.0)
    args = parser.parse_args()
    return args

def read_csv(filepath: str, delimiter=',') -> np.ndarray:
    return np.genfromtxt(filepath, delimiter=delimiter)

def reorder_dofs(dof_data: np.ndarray) -> np.ndarray:
    """
    Input: FR HIP X, FR HIP Y, FR KNEE, FL HIP X, FL HIP Y, FL KNEE, RR HIP X, RR HIP Y, RR KNEE, RL HIP X, RL HIP Y, RL KNEE
    Output: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    Input: (n_dof, ...)
    Output: (n_dof, ...)
    """
    fr_data = dof_data[:3, ...]
    fl_data = dof_data[3:6, ...]
    rr_data = dof_data[6:9, ...]
    rl_data = dof_data[9:12, ...]
    return np.concatenate([fl_data, fr_data, rl_data, rr_data], axis=0)

def change_dof_signs(dof_data: np.ndarray) -> np.ndarray:
    """
    C.f. Wanming's email: 
    Our URDFs are opposite so all joint pos, vel need to be flipped
    """
    return -dof_data

def parse_mocap_data(filepath: str) -> Tuple[np.ndarray, float]:
    """ 
    Input: Filepath to raw data 
    Output: (T, 19) array, each row is (body_pos, body_orn, dof_pos)
    """
    arr = read_csv(filepath)
    assert arr.shape[1] == 30
    dt = 0.04 # assume constant timestep
    body_pos = arr[:,0:3] # x, y, z
    body_orn = arr[:,3:6] # r, p, y
    r = Rotation.from_euler('xyz', body_orn)
    body_orn = r.as_quat()

    # Note that arr[:, 6:18] is DESIRED dof pos not actual; see Wanming's README
    dof_pos = reorder_dofs(arr[:,18:30].T).T # (FL, FR, RL, RR) sequence of (hip, thigh, calf) 
    dof_pos = change_dof_signs(dof_pos)
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

if __name__ == "__main__":
    args = parse_args()

    input_dir = pathlib.Path(args.input_dir)
    filepaths = input_dir.rglob("*.csv")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    output_filenames = []

    for filepath in filepaths:
        count += 1
        frame_data, dt = parse_mocap_data(str(filepath))
        n_timesteps = frame_data.shape[0]
        start_time = int(n_timesteps * args.start_time_frac)
        end_time = int(n_timesteps * args.end_time_frac)
        
        output_filename = filepath.relative_to(input_dir)
        output_filename = output_filename.parent / output_filename.stem
        output_filename = '_'.join(str(output_filename).split('/'))
        output_filename += '.txt'

        # Ensure no repeating filenames
        assert output_filename not in output_filenames
        output_filenames.append(output_filename)

        write_motion_data(
            output_dir / (output_filename), 
            frame_data[start_time: end_time], 
            dt, loop_mode='Clamp'
        )

    # Write metadata
    metadatas = []
    for fp in output_filenames:
        metadata = {'file': fp, 'weight': 1 / count}
        metadatas.append(metadata)
    import yaml
    with open(output_dir / 'dataset.yaml', 'w') as dataset_file:
        yaml.dump({'motions': metadatas}, dataset_file)

    