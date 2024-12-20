import argparse
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter


def pt_to_inch(w, h):
    return w / 72.27, h / 72.27


if __name__ == "__main__":
    # info
    print("+++ Plotting simulator performance")

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--env_img", type=str, required=True)
    arg_parser.add_argument("--num_envs_no_cam", type=int, nargs="+", required=True)
    arg_parser.add_argument("--total_fps_no_cam", type=int, nargs="+", required=True)
    arg_parser.add_argument("--vram_no_cam", type=int, nargs="+", required=True)
    arg_parser.add_argument("--num_envs_cam", type=int, nargs="+", required=True)
    arg_parser.add_argument("--total_fps_cam", type=int, nargs="+", required=True)
    arg_parser.add_argument("--vram_cam", type=int, nargs="+", required=True)
    arg_parser.add_argument("--fig_w", type=float, required=True)
    arg_parser.add_argument("--fig_h", type=float, required=True)
    arg_parser.add_argument("--fig_file", type=str, required=True)
    arg_parser.add_argument("--font_size", type=float, default=8)
    arg_parser.add_argument("--font_family", type=str, default="sans-serif")
    arg_parser.add_argument("--yaxis_offset", type=float, default=1.15)
    arg_parser.add_argument("--marker_size", type=int, default=4)
    arg_parser.add_argument("--dpi", type=int, default=1000)
    arg_parser.add_argument("--legend_vspace", type=float, default=0.25)

    args = arg_parser.parse_args()
    img_path: str = args.env_img
    num_envs_no_cam: List[int] = args.num_envs_no_cam
    total_fps_no_cam: List[int] = args.total_fps_no_cam
    vram_no_cam: List[int] = args.vram_no_cam
    num_envs_cam: List[int] = args.num_envs_cam
    total_fps_cam: List[int] = args.total_fps_cam
    vram_cam: List[int] = args.vram_cam
    fig_w: float = args.fig_w
    fig_h: float = args.fig_h
    fig_file: str = args.fig_file
    font_size: float = args.font_size
    font_family: str = args.font_family
    yaxis_offset: float = args.yaxis_offset
    marker_size: int = args.marker_size
    dpi: int = args.dpi
    legend_vspace: float = args.legend_vspace

    # check args
    assert len(num_envs_no_cam) == len(total_fps_no_cam) == len(vram_no_cam)
    assert len(num_envs_cam) == len(total_fps_cam) == len(vram_cam)
    print("---")
    print("image:", img_path)
    print("---")
    print("num_envs_no_cam:", num_envs_no_cam)
    print("total_fps_no_cam:", total_fps_no_cam)
    print("vram_no_cam:", vram_no_cam)
    print("---")
    print("num_envs_cam:", num_envs_cam)
    print("total_fps_cam:", total_fps_cam)
    print("vram_cam:", vram_cam)
    print("---")

    # load screenshot
    img: np.ndarray = np.asarray(Image.open(img_path))

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
        1, 3, figsize=pt_to_inch(fig_w, fig_h), constrained_layout=True
    )

    # image
    axs[0].imshow(img, interpolation="lanczos")
    axs[0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[0].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    legend_lines = None
    legend_labels = None
    for i in range(2):
        num_envs = None
        total_fps = None
        vram = None
        fps = None
        ylim_total_fps = None
        xlim = None
        xlabel = None
        if i == 0:
            num_envs = num_envs_no_cam
            total_fps = total_fps_no_cam
            vram = np.array(vram_no_cam) / (24 * 1024)
            fps = np.array(total_fps_no_cam) / np.array(num_envs_no_cam)
            ylim_total_fps = [0, 500_000]
            xlim = [0, 24576]
            xlabel = "Number of Environments without Camera"
        else:
            num_envs = num_envs_cam
            total_fps = total_fps_cam
            vram = np.array(vram_cam) / (24 * 1024)
            fps = np.array(total_fps_cam) / np.array(num_envs_cam)
            ylim_total_fps = [0, 5_000]
            xlim = [0, 1400]
            xlabel = "Number of Environments with Camera"

        twin_fps = axs[i + 1].twinx()
        twin_vram = axs[i + 1].twinx()
        twin_fps.spines.right.set_position(("axes", yaxis_offset))

        (l_total_fps,) = axs[i + 1].plot(
            num_envs,
            total_fps,
            color="tab:blue",
            marker=".",
            ms=marker_size,
            label="Total SPS",
        )
        (l_fps,) = twin_fps.plot(
            num_envs, fps, color="tab:orange", marker=".", ms=marker_size, label="SPS"
        )
        (l_vram,) = twin_vram.plot(
            num_envs,
            vram,
            color="tab:green",
            marker=".",
            ms=marker_size,
            label="VRAM",
        )

        axs[i + 1].yaxis.set_major_formatter(EngFormatter())
        axs[i + 1].xaxis.set_major_formatter(EngFormatter())
        axs[i + 1].grid(True)
        axs[i + 1].set_ylim(ylim_total_fps)
        axs[i + 1].set_xlim(xlim)
        axs[i + 1].set_xlabel(xlabel)
        axs[i + 1].tick_params(axis="y", colors=l_total_fps.get_color())

        twin_fps.tick_params(axis="y", colors=l_fps.get_color())
        twin_fps.set_ylim(0, 120)

        twin_vram.tick_params(axis="y", colors=l_vram.get_color())
        twin_vram.set_ylim(0, 1)

        if i == 0:
            twin_fps.axis("off")
            twin_vram.axis("off")
            legend_lines = [l_total_fps, l_fps, l_vram]
            legend_labels = [
                l_total_fps.get_label(),
                l_fps.get_label(),
                l_vram.get_label(),
            ]

    axs[0].legend(
        legend_lines,
        legend_labels,
        bbox_to_anchor=(0.0, -legend_vspace, 1.0, 1.0),
        loc="lower right",
        ncols=3,
        mode="expand",
        borderaxespad=0.0,
    )

    plt.savefig(fig_file, dpi=dpi)
    print(f"Plot saved to {fig_file}")
