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

    # Scatter plot of target velocity vs average followed velocity
    root_states = tensors['root_states']
    root_lin_vel = root_states[:, :, 7:10]
    dt = file.attrs['dt']

    task_state = tensors['task_state']
    # Sanity check: task state should be constant throughout episode
    assert np.all(task_state == task_state[:, 0:1, :])

    # Calculate target vel
    target_direction = task_state[:, 0, :3]
    target_speed = task_state[:, 0, -1:]
    target_vel = target_direction * target_speed
    # shape (N, 3)

    # Calculate average vel 
    start_root_pos = root_states[:, 0, :3]
    end_root_pos = root_states[:, -1, :3]
    total_time = dt * root_states.shape[1]
    average_vel = (end_root_pos - start_root_pos) / total_time
    # shape (N, 3)


    # Calculate dot-product
    dot_prod = np.sum(target_vel * average_vel, axis=-1)    
    norms = np.sum(target_vel ** 2, axis=-1)
    dot_prod = dot_prod / np.sum(target_vel ** 2, axis=-1)
    # shape (N,)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.set_tight_layout(True)

    # Plot histogram of fits
    def plot_histogram(ax):
        ax.hist(dot_prod, bins=20)
        ax.set_xlim(-4.0, 4.0)
        ax.set_title("Histogram of velocity-tracking error")
        ax.set_xlabel("Dot product of (target_vel, actual_vel)")

    # Plot scatterplot of target vs actual x-vel
    def plot_scatterplot(ax):
        ax.scatter(target_vel[:,0], average_vel[:,0],  label="x")
        ax.scatter(target_vel[:,1], average_vel[:,1], label="y")
        ax.scatter(target_vel[:,2], average_vel[:,2],  label="z")
        ax.set_title("Scatterplot of target and actual velocity")
        ax.set_xlabel("Target velocity (m/s)")
        ax.set_ylabel("Actual velocity (m/s)")
        ax.legend()

    plot_histogram(ax[0])
    plot_scatterplot(ax[1])
    
    fig.show()
    if args.output_filepath != "":
        fig.savefig(args.output_filepath)
    # Plot scatter-plot in x-dim      
    input("Press any key to exit...")

    
