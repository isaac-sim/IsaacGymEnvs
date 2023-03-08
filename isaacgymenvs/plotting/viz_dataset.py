import argparse
import h5py
import numpy as np
import pathlib

from isaacgymenvs.utilities.viz_util import plot_motion_data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize expert dataset")
    parser.add_argument("-i", "--input-filepath", type=str)
    parser.add_argument('-o', "--output-filepath", type=str, default="")
    parser.add_argument("-t", "--trajectory-idx", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    file = h5py.File(args.input_filepath)
    for key, value in file.attrs.items():
        print(f"{key}: {value}")
    tensors = {}
    for name in file.keys():
        size = file[name].attrs['size']
        # Remove last index due to reset
        tensors[name] = file[name][:size][:, :-1, :]
        print(name, size)
        print(tensors[name].shape)

    ts = np.arange(file.attrs['max_episode_length']-1) * file.attrs['dt']
    
    # Visualize one trajectory from the dataset
    traj_idx = args.trajectory_idx
    root_states = tensors['root_states'][traj_idx]
    dof_pos = tensors['dof_pos'][traj_idx]
    dof_vel = tensors['dof_vel'][traj_idx]
    fig, ax = plot_motion_data(ts, root_states, dof_pos, dof_vel)
    filename = pathlib.Path(args.input_filepath).stem
    fig.suptitle(filename, size=30)
    fig.show()    

    if args.output_filepath != "":
        fig.savefig(args.output_filepath)
    input("Press any key to exit...")