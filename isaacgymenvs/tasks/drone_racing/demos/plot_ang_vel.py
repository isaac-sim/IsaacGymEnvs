import argparse
import warnings
from typing import Dict, Any, List

import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def pt_to_inch(w, h):
    return w / 72.27, h / 72.27


if __name__ == "__main__":
    # info
    print("+++ Plotting desired and controlled angular velocity")

    # Suppress torch.load warning
    warnings.filterwarnings("ignore", category=FutureWarning)

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_file", type=str, required=True)
    arg_parser.add_argument("--episode_id", type=int, required=True)
    arg_parser.add_argument("--ctrl_dt", type=float, required=True)
    arg_parser.add_argument("--fig_w", type=float, required=True)
    arg_parser.add_argument("--fig_h", type=float, required=True)
    arg_parser.add_argument("--fig_file", type=str, required=True)
    arg_parser.add_argument("--ang_vel_max", type=float, default=14)
    arg_parser.add_argument("--font_size", type=int, default=10)
    arg_parser.add_argument("--font_family", type=str, default="sans-serif")
    arg_parser.add_argument("--legend_vspace", type=float, default=0.05)

    args = arg_parser.parse_args()
    log_file: str = args.log_file
    episode_id: int = args.episode_id
    ctrl_dt: float = args.ctrl_dt
    fig_w: float = args.fig_w
    fig_h: float = args.fig_h
    fig_file: str = args.fig_file
    ang_vel_max: float = args.ang_vel_max
    font_size: int = args.font_size
    font_family: str = args.font_family
    legend_vspace: float = args.legend_vspace

    # load log and get data
    ep_dict: Dict[str, Any] = torch.load(log_file)
    des_ang_vel_list: List[torch.Tensor] = ep_dict[f"ep_{episode_id}"][
        "ang_vel_des_b_frd"
    ]
    ang_vel_list: List[torch.Tensor] = ep_dict[f"ep_{episode_id}"]["ang_vel_b_frd"]

    # pre-process data
    des_ang_vel = torch.stack(des_ang_vel_list).flatten(0, 1)
    ang_vel = torch.stack(ang_vel_list).flatten(0, 1)
    assert des_ang_vel.shape == ang_vel.shape

    # data to plot
    t = (torch.arange(des_ang_vel.shape[0]) * ctrl_dt).numpy()
    des_ang_vel_x = des_ang_vel[:, 0].numpy()
    des_ang_vel_y = des_ang_vel[:, 1].numpy()
    des_ang_vel_z = des_ang_vel[:, 2].numpy()
    ang_vel_x = ang_vel[:, 0].numpy()
    ang_vel_y = ang_vel[:, 1].numpy()
    ang_vel_z = ang_vel[:, 2].numpy()

    # plot
    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": font_family,
            "font.sans-serif": "Arial",
            "font.serif": "Times New Roman",
        },
    )
    fig, axs = plt.subplots(
        3, 1, figsize=pt_to_inch(fig_w, fig_h), sharex="col", constrained_layout=True
    )
    axs[-1].set_xlabel("Time (s)")
    for i in range(3):
        desired = None
        measured = None
        label = None
        if i == 0:
            desired = des_ang_vel_x
            measured = ang_vel_x
            label = "X (rad/s)"
        elif i == 1:
            desired = des_ang_vel_y
            measured = ang_vel_y
            label = "Y (rad/s)"

        elif i == 2:
            desired = des_ang_vel_z
            measured = ang_vel_z
            label = "Z (rad/s)"

        axs[i].plot(t, desired, label="Desired Angular Velocity")
        axs[i].plot(t, measured, label="Measured Angular Velocity")
        axs[i].grid(True)
        axs[i].set_xlim([t[0], t[-1]])
        axs[i].set_ylim([-ang_vel_max, ang_vel_max])
        axs[i].set_ylabel(label)
        axs[i].minorticks_on()
        axs[i].xaxis.set_minor_locator(MultipleLocator(0.1))

        if i == 0:
            axs[i].legend(
                bbox_to_anchor=(0.0, 1 + legend_vspace, 1.0, 1.0),
                loc="lower right",
                ncols=2,
                borderaxespad=0.0,
            )

    plt.savefig(fig_file)
    print(f"Plot saved to {fig_file}")
