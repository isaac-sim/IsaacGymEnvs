import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter


def pt_to_inch(w, h):
    return w / 72.27, h / 72.27


if __name__ == "__main__":
    # info
    print("+++ Plotting train log")

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--ep_len_csv", type=str, required=True)
    arg_parser.add_argument("--rew_csv", type=str, required=True)
    arg_parser.add_argument("--rew_col_csv", type=str, default="")
    arg_parser.add_argument("--rew_wp_csv", type=str, default="")
    arg_parser.add_argument("--fig_w", type=float, required=True)
    arg_parser.add_argument("--fig_h", type=float, required=True)
    arg_parser.add_argument("--fig_file", type=str, required=True)
    arg_parser.add_argument("--font_size", type=float, default=6)
    arg_parser.add_argument("--font_family", type=str, default="sans-serif")
    arg_parser.add_argument("--xlim_low", type=float, default=0)
    arg_parser.add_argument("--xlim_high", type=float, required=True)
    arg_parser.add_argument("--lwidth", type=float, default=0.5)

    args = arg_parser.parse_args()
    ep_len_csv: str = args.ep_len_csv
    rew_csv: str = args.rew_csv
    rew_col_csv: str = args.rew_col_csv
    rew_wp_csv: str = args.rew_wp_csv
    fig_w: float = args.fig_w
    fig_h: float = args.fig_h
    fig_file: str = args.fig_file
    font_size: float = args.font_size
    font_family: str = args.font_family
    xlim_low: float = args.xlim_low
    xlim_high: float = args.xlim_high
    lwidth: float = args.lwidth

    # load csv
    ep_len_df = pd.read_csv(ep_len_csv)
    rew_df = pd.read_csv(rew_csv)
    rew_col_df = None
    rew_wp_df = None
    if rew_col_csv != "":
        rew_col_df = pd.read_csv(rew_col_csv)
    if rew_wp_csv != "":
        rew_wp_df = pd.read_csv(rew_wp_csv)

    # extract data
    global_step = np.asarray(ep_len_df["global_step"])
    ep_len = np.asarray(ep_len_df.iloc[:, 4])
    rew = np.asarray(rew_df.iloc[:, 4])
    rew_col = None
    rew_wp = None
    if rew_col_csv != "":
        rew_col = rew_col_df.iloc[:, 4]
    if rew_wp_csv != "":
        rew_wp = rew_wp_df.iloc[:, 4]

    assert len(ep_len) == len(rew) == len(ep_len)

    # plot
    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": font_family,
            "font.sans-serif": "Arial",
            "font.serif": "Times New Roman",
        },
    )

    if rew_col_csv == "" and rew_col_csv == "":
        print("plotting only ep length and total reward")
        fig, axs = plt.subplots(
            1, 2, figsize=pt_to_inch(fig_w, fig_h), constrained_layout=True
        )

        for i in [0, 1]:
            item = None
            label = None
            if i == 0:
                item = rew
                label = "Mean Episode Reward"
            else:
                item = ep_len
                label = "Mean Episode Length"
            axs[i].plot(global_step, item)
            axs[i].xaxis.set_major_formatter(EngFormatter())
            axs[i].grid(True)
            axs[i].set_ylabel(label)
            axs[i].set_xlabel("Number of Total Steps")
            axs[i].set_xlim([xlim_low, xlim_high])

    elif rew_col_csv != "" or rew_col_csv != "":
        print("plotting ep len, total rew, collision rew, wp rew")
        fig, axs = plt.subplots(
            4,
            1,
            figsize=pt_to_inch(fig_w, fig_h),
            constrained_layout=True,
            sharex="col",
        )

        for i in [0, 1, 2, 3]:
            item = None
            if i == 0:
                item = ep_len
            if i == 1:
                item = rew
            if i == 2:
                item = rew_col
                axs[i].set_ylim([-10, 0])
            if i == 3:
                item = rew_wp
                axs[i].set_ylim([0, 10])
            axs[i].plot(global_step, item, linewidth=lwidth)
            axs[i].xaxis.set_major_formatter(EngFormatter())
            axs[i].grid(True)
            axs[i].set_xlim([xlim_low, xlim_high])
            axs[i].yaxis.set_label_coords(-0.1, 0.5)

        axs[0].set_ylabel("Mean Episode Length")
        axs[1].set_ylabel("Mean Total Reward")
        axs[2].set_ylabel("Mean Collision Reward")
        axs[3].set_ylabel("Mean Waypoint Reward")
        axs[3].set_xlabel("Number of Total Steps")

    else:
        raise ValueError

    plt.savefig(fig_file)
    print(f"Plot saved to {fig_file}")
