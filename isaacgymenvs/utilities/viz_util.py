import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, List, Tuple

def line_plot(
    ax: plt.Axes, 
    x: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    fontsize: int = 20,
    xlabel: str = "", 
    ylabel: str = "", 
    labels: Optional[List[str]] = None
) -> None:
    """ Utility function to do a line plot """
    ax.plot(x, y)
    ax.set_title(title, size = fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels is not None:
        ax.legend(labels)

def plot_motion_data(
        ts: np.ndarray, 
        root_states: np.ndarray, 
        dof_pos: np.ndarray, 
        dof_vel: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """ Plots a single motion trajectory

    args:
        ts: [T,]-shape array of times, T = num trajectory steps
        root_states: [T, 13]-shape tensor of root states
        dof_pos: [T, d]-shape tensor of DoF positions
        dof_vel: [T, d]-shape tensor of DoF velocities
    """
    fig, ax = plt.subplots(2, 3, figsize=(40, 20))
    fig.set_tight_layout(True)

    # Body pos 
    body_pos = root_states[:, :3]
    line_plot(ax[0][0], ts, body_pos, 
            title="Body Pos", 
            xlabel="Time (s)",
            ylabel="Position (m)",
            labels = ["x", "y", "z"]
    )

    # Body orn
    body_orn = root_states[:, 3:7]
    line_plot(ax[0][1], ts, body_orn, 
            title="Body Orn",  
            xlabel="Time (s)", 
            labels = ["qx", "qy", "qz", "qw"]
    )

    # Dof pos
    line_plot(ax[0][2], ts, dof_pos, 
            title="Dof Pos",  
            xlabel="Time (s)",
            ylabel="Joint pos (rad)"
    )

    # Body ang vel
    body_lin_vel = root_states[:, 7:10]
    line_plot(ax[1][0], ts, body_lin_vel, 
            title="Body Lin Vel",  
            xlabel="Time (s)",
            ylabel="Velocity (m/s)",
            labels=["dx", "dy", "dz"]
    )

    # Body ang vel
    body_ang_vel = root_states[:, 10:]
    line_plot(ax[1][1], ts, body_ang_vel, 
            title="Body Ang Vel",  
            xlabel="Time (s)", 
            ylabel="Ang vel (rad/s)",
            labels=["dR", "dP", "dY"]
    )

    # Dof vel
    line_plot(ax[1][2], ts, dof_vel, 
            title="Dof Vel",  
            xlabel="Time (s)",
            ylabel="Joint vel (rad/s)"
              
    ) 

    return fig, ax 