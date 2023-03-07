import argparse
import h5py
import numpy as np
import pathlib

from isaacgymenvs.utilities.viz_util import plot_motion_data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize expert dataset")
    parser.add_argument("-f", "--filepath", type=str)
    parser.add_argument("-t", "--trajectory-idx", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    file = h5py.File(args.filepath)
    for key, value in file.attrs.items():
        print(f"{key}: {value}")
    for name in file.keys():
        print(name)
        print(file[name].shape)
        print(file[name].attrs['size'])

    ts = np.arange(file.attrs['max_episode_length']) * file.attrs['dt']
    
    # Visualize one trajectory from the dataset
    traj_idx = args.trajectory_idx
    root_states = file['root_states'][traj_idx]
    dof_pos = file['dof_pos'][traj_idx]
    dof_vel = file['dof_vel'][traj_idx]
    fig, ax = plot_motion_data(ts, root_states, dof_pos, dof_vel)
    filename = pathlib.Path(args.filepath).stem
    fig.suptitle(filename, size=30)
    fig.show()    
    input("Press any key to exit...")