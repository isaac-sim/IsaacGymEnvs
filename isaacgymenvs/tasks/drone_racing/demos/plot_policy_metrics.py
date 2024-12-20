import argparse
import os.path
import warnings
from typing import List, Dict, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def pt_to_inch(w, h):
    return w / 72.27, h / 72.27


def main():
    # info
    print("+++ Plotting policy metrics vs. task complexity")

    # Suppress torch.load warning
    warnings.filterwarnings("ignore", category=FutureWarning)

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dirs",
        type=str,
        nargs="+",
        required=True,
        help="experiment directories given in order of simple to hard",
    )
    arg_parser.add_argument("--font_size", type=float, default=6)
    arg_parser.add_argument("--font_family", type=str, default="serif")
    arg_parser.add_argument("--fig_w", type=float, required=True)
    arg_parser.add_argument("--fig_h", type=float, required=True)
    arg_parser.add_argument("--fig_file", type=str, required=True)
    arg_parser.add_argument("--dpi", type=int, default=1000)

    args = arg_parser.parse_args()
    exp_dirs: List[str] = args.exp_dirs
    font_size: float = args.font_size
    font_family: str = args.font_family
    fig_w: float = args.fig_w
    fig_h: float = args.fig_h
    fig_file: str = args.fig_file
    dpi: int = args.dpi

    # aggregate metrics
    success_rates: List[float] = []  # length will be num_exps
    safety_margins: List[List[float]] = []
    avg_lin_speeds: List[List[float]] = []
    max_lin_speeds: List[List[float]] = []
    avg_ang_speeds: List[List[float]] = []
    max_ang_speeds: List[List[float]] = []
    avg_avg_rotor_cmds: List[List[float]] = []
    max_avg_rotor_cmds: List[List[float]] = []
    for exp_dir in exp_dirs:
        # load metrics file
        exp_metrics: Dict[str, Any] = torch.load(os.path.join(exp_dir, "metrics.pt"))

        # print data
        sr = exp_metrics["num_finishes"] / (
            exp_metrics["num_finishes"]
            + exp_metrics["num_crashes"]
            + exp_metrics["num_timeouts"]
        )
        print(exp_dir)
        print(f"- sr = {sr}")
        print(f"- avg_lin_speed = {np.mean(exp_metrics['avg_lin_speeds'])}")
        print(f"- max_lin_speed = {np.max(exp_metrics['max_lin_speeds'])}")
        print(f"- avg_ang_speed = {np.mean(exp_metrics['avg_ang_speeds'])}")
        print(f"- max_ang_speed = {np.max(exp_metrics['max_ang_speeds'])}")
        print(f"- avg_avg_rotor_cmd = {np.mean(exp_metrics['avg_avg_rotor_cmds'])}")
        print(f"- max_avg_rotor_cmd = {np.max(exp_metrics['max_avg_rotor_cmds'])}")

        # append info
        success_rates.append(sr)
        safety_margins.append(exp_metrics["min_safety_margins"])
        avg_lin_speeds.append(exp_metrics["avg_lin_speeds"])
        max_lin_speeds.append(exp_metrics["max_lin_speeds"])
        avg_ang_speeds.append(exp_metrics["avg_ang_speeds"])
        max_ang_speeds.append(exp_metrics["max_ang_speeds"])
        avg_avg_rotor_cmds.append(exp_metrics["avg_avg_rotor_cmds"])
        max_avg_rotor_cmds.append(exp_metrics["max_avg_rotor_cmds"])

    # plot data
    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": font_family,
            "font.sans-serif": "Arial",
            "font.serif": "Times New Roman",
        },
    )
    fig, axs = plt.subplots(
        2, 2, figsize=pt_to_inch(fig_w, fig_h), constrained_layout=True, sharex="col"
    )

    # safety margin and success rate
    axs[0][0].violinplot(safety_margins, showmeans=True)
    axs[0][0].plot(np.arange(len(success_rates)) + 1, success_rates, marker=".")
    axs[0][0].grid(True)
    legend_elements = [
        Patch(facecolor="tab:blue", label="Safety Margin"),
        Patch(facecolor="tab:orange", label="Success Rate"),
    ]
    axs[0][0].legend(handles=legend_elements)
    axs[0][0].set_ylim([0, 1.2])
    axs[0][0].set_ylabel("Safety Margin (m) and Success Rate")
    axs[0][0].yaxis.set_label_coords(-0.175, 0.5)

    # rotor commands
    axs[0][1].violinplot(
        avg_avg_rotor_cmds,
        showmeans=True,
        positions=np.arange(1, len(avg_avg_rotor_cmds) + 1) - 0.125,
    )
    axs[0][1].violinplot(
        max_avg_rotor_cmds,
        showmeans=True,
        positions=np.arange(1, len(max_avg_rotor_cmds) + 1) + 0.125,
    )
    axs[0][1].grid(True)
    legend_elements = [
        Patch(facecolor="tab:blue", label="Mean"),
        Patch(facecolor="tab:orange", label="Max"),
    ]
    axs[0][1].legend(handles=legend_elements)
    axs[0][1].set_ylabel("Average Motor Commands")
    axs[0][1].yaxis.set_label_coords(-0.175, 0.5)

    # linear speed
    axs[1][0].violinplot(
        avg_lin_speeds,
        showmeans=True,
        positions=np.arange(1, len(avg_lin_speeds) + 1) - 0.125,
    )
    axs[1][0].violinplot(
        max_lin_speeds,
        showmeans=True,
        positions=np.arange(1, len(max_lin_speeds) + 1) + 0.125,
    )
    axs[1][0].grid(True)
    axs[1][0].set_xlabel("Difficulty Level")
    legend_elements = [
        Patch(facecolor="tab:blue", label="Mean"),
        Patch(facecolor="tab:orange", label="Max"),
    ]
    axs[1][0].legend(handles=legend_elements)
    axs[1][0].set_ylabel("Linear Speed (m/s)")
    axs[1][0].yaxis.set_label_coords(-0.175, 0.5)

    # angular speed
    axs[1][1].violinplot(
        avg_ang_speeds,
        showmeans=True,
        positions=np.arange(1, len(avg_ang_speeds) + 1) - 0.125,
    )
    axs[1][1].violinplot(
        max_ang_speeds,
        showmeans=True,
        positions=np.arange(1, len(max_ang_speeds) + 1) + 0.125,
    )
    axs[1][1].grid(True)
    axs[1][1].set_xlabel("Difficulty Level")
    legend_elements = [
        Patch(facecolor="tab:blue", label="Mean"),
        Patch(facecolor="tab:orange", label="Max"),
    ]
    axs[1][1].legend(handles=legend_elements)
    axs[1][1].set_ylabel("Angular Speed (rad/s)")
    axs[1][1].yaxis.set_label_coords(-0.175, 0.5)

    plt.savefig(fig_file, dpi=dpi)
    print(f"Plot saved to {fig_file}")


if __name__ == "__main__":
    main()
