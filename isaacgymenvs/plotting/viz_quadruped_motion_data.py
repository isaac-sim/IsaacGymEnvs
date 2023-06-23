
import isaacgym 
import torch 
import pathlib
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt 

from isaacgymenvs.utilities.quadruped_motion_data import MotionData, MotionLib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize quadruped motion data")
    parser.add_argument("--filepath", type=str,  default='/home/robohike/A1_IsaacGym/IsaacGymEnvs/isaacgymenvs/data/motions/quadruped/mania_pos/motion7.txt')
    parser.add_argument("-s", "--start-time-frac", type=float, help="Start time as a fraction. E.g. 0.2 = start from 20% of the way in", default = 0.0)
    parser.add_argument("-e", "--end-time-frac", type=float, help="End time as a fraction. E.g. 0.8 = end at 80% of the way in", default = 1.0)
    parser.add_argument("-n", "--num-timesteps", type=int, help="Number of interpolation timesteps to use within (start_time, end_time)", default=1000)
    args = parser.parse_args()
    return args

def line_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str, fontsize: int = 20):
    ax.plot(x, y)
    ax.set_title(title, size= fontsize)

def plot_motion_data(ts, frames, frame_vels) -> plt.Axes:
    fig, ax = plt.subplots(2, 3, figsize=(40, 20))

    # Body pos 
    body_pos = frames[:, :3]
    line_plot(ax[0][0], ts, body_pos, title="Body Pos")

    # Body orn
    body_orn = frames[:, 3:7]
    line_plot(ax[0][1], ts, body_orn, title="Body Orn")

    # Dof pos 
    dof_pos = frames[:, 7:]
    line_plot(ax[0][2], ts, dof_pos, title="Dof Pos")

    # Body lin vel
    body_lin_vel = frame_vels[:, :3]
    line_plot(ax[1][0], ts, body_lin_vel, title="Body Lin Vel")

    # Body ang vel
    body_ang_vel = frame_vels[:, 3:6]
    line_plot(ax[1][1], ts, body_ang_vel, title="Body Ang Vel")

    # Dof vel
    dof_vel = frame_vels[:, 6:]
    line_plot(ax[1][2], ts, dof_vel, title="Dof Vel") 

    return fig, ax 

if __name__ == "__main__":

    args = parse_args()
    motion_data = MotionData(args.filepath)

    total_time = motion_data.get_duration()
    start_time = args.start_time_frac * total_time 
    end_time = args.end_time_frac * total_time
    n_timesteps = args.num_timesteps

    ts = np.linspace(start_time, end_time, n_timesteps)
    frames = np.zeros((n_timesteps, 19))
    frame_vels = np.zeros((n_timesteps, 18))

    for i, t in enumerate(ts):
        frames[i] = motion_data.calc_frame(t)
        frame_vels[i] = motion_data.calc_frame_vel(t)

    fig, ax = plot_motion_data(ts, frames, frame_vels)
    filename = pathlib.Path(args.filepath).stem
    fig.suptitle(filename, size=30)
    fig.savefig('sine_wave.png')
    fig.show()    
    # input("Press any key to exit...")