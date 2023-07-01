"""Run groups of experiments, hyperparameter sweeps, etc."""

import argparse
import os
import subprocess
import sys
import time
from os.path import join


def add_os_parallelism_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--num_gpus", default=1, type=int, help="How many local GPUs to use")
    parser.add_argument("--max_parallel", default=4, type=int, help="Maximum simultaneous experiments")
    parser.add_argument(
        "--experiments_per_gpu",
        default=-1,
        type=int,
        help="How many experiments can we squeeze on a single GPU. "
        "Specify this option if and only if you are using launcher to run several experiments using OS-level"
        "parallelism (--backend=processes)."
        "In any other case use default value (-1) for not altering CUDA_VISIBLE_DEVICES at all."
        "This will allow your experiments to use all GPUs available (as many as --num_gpu allows)"
        "Helpful when e.g. you are running a single big PBT experiment.",
    )
    return parser


def ensure_dir_exists(path) -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def run(run_description, args):
    experiments = run_description.experiments
    max_parallel = args.max_parallel

    print("Starting processes with base cmds: %r", [e.cmd for e in experiments])
    print(f"Max parallel processes is {max_parallel}")
    print(f"Monitor log files using\n\n\ttail -f train_dir/{run_description.run_name}/**/**/sf_log.txt\n\n")

    processes = []
    processes_per_gpu = {g: [] for g in range(args.num_gpus)}

    experiments = run_description.generate_experiments(args.train_dir)
    next_experiment = next(experiments, None)

    def find_least_busy_gpu():
        least_busy_gpu = None
        gpu_available_processes = 0

        for gpu_id in range(args.num_gpus):
            available_processes = args.experiments_per_gpu - len(processes_per_gpu[gpu_id])
            if available_processes > gpu_available_processes:
                gpu_available_processes = available_processes
                least_busy_gpu = gpu_id

        return least_busy_gpu, gpu_available_processes

    def can_squeeze_another_process():
        if len(processes) >= max_parallel:
            return False

        if args.experiments_per_gpu > 0:
            least_busy_gpu, gpu_available_processes = find_least_busy_gpu()
            if gpu_available_processes <= 0:
                return False

        return True

    failed_processes = []
    last_log_time = 0
    log_interval = 3  # seconds

    while len(processes) > 0 or next_experiment is not None:
        while can_squeeze_another_process() and next_experiment is not None:
            cmd, name, root_dir, exp_env_vars = next_experiment

            cmd_tokens = cmd.split(" ")

            # workaround to make sure we're running the correct python executable from our virtual env
            if cmd_tokens[0].startswith("python"):
                cmd_tokens[0] = sys.executable
                print(f"Using Python executable {cmd_tokens[0]}")

            ensure_dir_exists(join(args.train_dir, root_dir))

            envvars = os.environ.copy()

            best_gpu = None
            if args.experiments_per_gpu > 0:
                best_gpu, best_gpu_available_processes = find_least_busy_gpu()
                print(
                    f"The least busy gpu is {best_gpu} where we can run {best_gpu_available_processes} more processes",
                )
                envvars["CUDA_VISIBLE_DEVICES"] = f"{best_gpu}"

            print(f"Starting process {cmd_tokens}")

            if exp_env_vars is not None:
                for key, value in exp_env_vars.items():
                    print(f"Adding env variable {key} {value}")
                    envvars[str(key)] = str(value)

            process = subprocess.Popen(cmd_tokens, stdout=None, stderr=None, env=envvars)
            process.gpu_id = best_gpu
            process.proc_cmd = cmd

            processes.append(process)

            if process.gpu_id is not None:
                processes_per_gpu[process.gpu_id].append(process.proc_cmd)

            print(f"Started process {process.proc_cmd} GPU {process.gpu_id}")
            print(f"Waiting for {args.pause_between} seconds before starting next process")
            time.sleep(args.pause_between)

            next_experiment = next(experiments, None)

        remaining_processes = []
        for process in processes:
            if process.poll() is None:
                remaining_processes.append(process)
                continue
            else:
                if process.gpu_id is not None:
                    processes_per_gpu[process.gpu_id].remove(process.proc_cmd)
                print(f"Process finished {process.proc_cmd}, {process.returncode}")
                if process.returncode != 0:
                    failed_processes.append((process.proc_cmd, process.pid, process.returncode))
                    print(f"WARNING: RETURN CODE IS {process.returncode}")

        processes = remaining_processes

        if time.time() - last_log_time > log_interval:
            if failed_processes:
                print(f"Failed processes:", ", ".join([f"PID: {p[1]} code: {p[2]}" for p in failed_processes]))
            last_log_time = time.time()

        time.sleep(0.1)

    print("Done!")

    return 0
