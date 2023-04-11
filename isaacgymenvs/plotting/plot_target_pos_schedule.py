import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize expert dataset")
    parser.add_argument("-i", "--input-filepath", type=str)
    parser.add_argument('-o', "--output-filepath", type=str, default="")
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

    root_states = tensors['root_states']
    dt = file.attrs['dt']
    task_state = tensors['task_state']

    traj_idx = 0
    current_position = task_state[traj_idx, :, 5:8]
    goal_position = task_state[traj_idx, :, 8:]
    ts = dt * np.arange(current_position.shape[0])

    def plot_target_vs_actual_position_trajectory(ax):    
        ax.plot(ts, current_position[:,0],  label="Actual x")
        ax.plot(ts, current_position[:,1],  label="Actual y")
        ax.plot(ts, goal_position[:,0],  label="Target x")
        ax.plot(ts, goal_position[:,1],  label="Target y")
        ax.legend()

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_tight_layout(True)
    plot_target_vs_actual_position_trajectory(ax)
    
    fig.show()
    if args.output_filepath != "":
        fig.savefig(args.output_filepath)
    # Plot scatter-plot in x-dim      
    input("Press any key to exit...")

    
