import argparse

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def pt_to_inch(w, h):
    return w / 72.27, h / 72.27


if __name__ == "__main__":
    # info
    print("+++ Plotting trajectories on Split-S and Turns")

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--img_splits", type=str, required=True)
    arg_parser.add_argument("--img_turns", type=str, required=True)
    arg_parser.add_argument("--max_v", type=float, required=True)
    arg_parser.add_argument("--colormap", type=str, required=True)
    arg_parser.add_argument("--fig_w", type=float, required=True)
    arg_parser.add_argument("--fig_h", type=float, required=True)
    arg_parser.add_argument("--fig_file", type=str, required=True)
    arg_parser.add_argument("--font_size", type=float, default=8)
    arg_parser.add_argument("--font_family", type=str, default="sans-serif")
    arg_parser.add_argument("--dpi", type=int, default=1000)
    arg_parser.add_argument("--cbar_aspect", type=float, default=45)

    args = arg_parser.parse_args()
    img_splits_path: str = args.img_splits
    img_turns_path: str = args.img_turns
    max_v: float = args.max_v
    colormap: str = args.colormap
    fig_w: float = args.fig_w
    fig_h: float = args.fig_h
    fig_file: str = args.fig_file
    font_size: float = args.font_size
    font_family: str = args.font_family
    dpi: float = args.dpi
    cbar_aspect: float = args.cbar_aspect

    # load images
    img_splits: np.ndarray = np.asarray(Image.open(img_splits_path))
    img_turns: np.ndarray = np.asarray(Image.open(img_turns_path))

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
        1, 2, figsize=pt_to_inch(fig_w, fig_h), constrained_layout=True
    )

    for i in [0, 1]:
        img = None
        label = None
        if i == 0:
            img = img_splits
            label = "Split-S"
        else:
            img = img_turns
            label = "Turns"

        im = axs[i].imshow(img, interpolation="lanczos")
        axs[i].tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )
        axs[i].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[i].set_xlabel(label)

        norm = Normalize(vmin=0, vmax=25)  # Define the color range
        sm = ScalarMappable(norm=norm, cmap="plasma")
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axs[i],
            orientation="horizontal",
            location="top",
            aspect=cbar_aspect,
            shrink=1.0,
        )

    plt.savefig(fig_file, dpi=dpi)
    print(f"Plot saved to {fig_file}")
