"""
Run many experiments with NGC: hyperparameter sweeps, etc.
This isn't production code, but feel free to use as an example for your NGC setup.

"""

import time
from multiprocessing.pool import ThreadPool
from subprocess import PIPE, Popen

from isaacgymenvs.pbt.launcher.run_slurm import str2bool


def add_ngc_args(parser):
    parser.add_argument(
        "--ngc_job_template",
        default=None,
        type=str,
        help="NGC command line template, specifying instance type, docker container, etc.",
    )
    parser.add_argument(
        "--ngc_print_only", default=False, type=str2bool, help="Just print commands to the console without executing"
    )

    parser.set_defaults(pause_between=0)
    return parser


def run_ngc(run_description, args):
    pause_between = args.pause_between
    experiments = run_description.experiments

    print(f"Starting processes with base cmds: {[e.cmd for e in experiments]}")

    if args.ngc_job_template is not None:
        with open(args.ngc_job_template, "r") as template_file:
            ngc_template = template_file.read()

    ngc_template = ngc_template.replace("\\", " ")
    ngc_template = " ".join(ngc_template.split())

    print(f"NGC template: {ngc_template}")
    experiments = run_description.generate_experiments(args.train_dir, makedirs=False)
    experiments = list(experiments)
    print(f"{len(experiments)} experiments to run")

    def launch_experiment(experiment_idx, experiment_):
        time.sleep(experiment_idx * 0.1)

        cmd, name, *_ = experiment_

        job_name = name
        print(f"Job name: {job_name}")

        ngc_job_cmd = ngc_template.replace("{{ name }}", job_name).replace("{{ experiment_cmd }}", cmd)

        print(f"Executing {ngc_job_cmd}")

        if not args.ngc_print_only:
            process = Popen(ngc_job_cmd, stdout=PIPE, shell=True)
            output, err = process.communicate()
            exit_code = process.wait()
            print(f"Output: {output}, err: {err}, exit code: {exit_code}")

        time.sleep(pause_between)

    pool_size = 1 if pause_between > 0 else min(10, len(experiments))
    with ThreadPool(pool_size) as p:
        p.starmap(launch_experiment, enumerate(experiments))

    print("Done!")
    return 0
