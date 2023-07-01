import argparse
import importlib
import sys

from isaacgymenvs.pbt.launcher.run_ngc import add_ngc_args, run_ngc
from isaacgymenvs.pbt.launcher.run_processes import add_os_parallelism_args, run
from isaacgymenvs.pbt.launcher.run_slurm import add_slurm_args, run_slurm


def launcher_argparser(args) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./train_dir", type=str, help="Directory for sub-experiments")
    parser.add_argument(
        "--run",
        default=None,
        type=str,
        help="Name of the python module that describes the run, e.g. sf_examples.vizdoom.experiments.paper_doom_all_basic_envs.py "
        "Run module must be importable in your Python environment. It must define a global variable RUN_DESCRIPTION (see existing run modules for examples).",
    )
    parser.add_argument(
        "--backend",
        default="processes",
        choices=["processes", "slurm", "ngc"],
        help="Launcher backend, use OS multiprocessing by default",
    )
    parser.add_argument("--pause_between", default=1, type=int, help="Pause in seconds between processes")
    parser.add_argument(
        "--experiment_suffix", default="", type=str, help="Append this to the name of the experiment dir"
    )

    partial_cfg, _ = parser.parse_known_args(args)

    if partial_cfg.backend == "slurm":
        parser = add_slurm_args(parser)
    elif partial_cfg.backend == "ngc":
        parser = add_ngc_args(parser)
    elif partial_cfg.backend == "processes":
        parser = add_os_parallelism_args(parser)
    else:
        raise ValueError(f"Unknown backend: {partial_cfg.backend}")

    return parser


def parse_args():
    args = launcher_argparser(sys.argv[1:]).parse_args(sys.argv[1:])
    return args


def main():
    launcher_cfg = parse_args()

    try:
        # assuming we're given the full name of the module
        run_module = importlib.import_module(f"{launcher_cfg.run}")
    except ImportError as exc:
        print(f"Could not import the run module {exc}")
        return 1

    run_description = run_module.RUN_DESCRIPTION
    run_description.experiment_suffix = launcher_cfg.experiment_suffix

    if launcher_cfg.backend == "processes":
        run(run_description, launcher_cfg)
    elif launcher_cfg.backend == "slurm":
        run_slurm(run_description, launcher_cfg)
    elif launcher_cfg.backend == "ngc":
        run_ngc(run_description, launcher_cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
