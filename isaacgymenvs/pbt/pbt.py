# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import os
import random
import shutil
import sys
import time
from os.path import join
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from rl_games.algos_torch.torch_ext import safe_filesystem_op, safe_save
from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.pbt.mutation import mutate
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import flatten_dict, project_tmp_dir, safe_ensure_dir_exists


# i.e. value for target objective when it is not known
_UNINITIALIZED_VALUE = float(-1e9)


def _checkpnt_name(iteration):
    return f"{iteration:06d}.yaml"


def _model_checkpnt_name(iteration):
    return f"{iteration:06d}.pth"


def _flatten_params(params: Dict, prefix="", separator=".") -> Dict:
    all_params = flatten_dict(params, prefix, separator)
    return all_params


def _filter_params(params: Dict, params_to_mutate: Dict) -> Dict:
    filtered_params = dict()
    for key, value in params.items():
        if key in params_to_mutate:
            if isinstance(value, str):
                try:
                    # trying to convert values such as "1e-4" to floats because yaml fails to recognize them as such
                    float_value = float(value)
                    value = float_value
                except ValueError:
                    pass

            filtered_params[key] = value
    return filtered_params


class PbtParams:
    def __init__(self, cfg: DictConfig):
        params: Dict = omegaconf_to_dict(cfg)

        pbt_params = params["pbt"]
        self.replace_fraction_best = pbt_params["replace_fraction_best"]
        self.replace_fraction_worst = pbt_params["replace_fraction_worst"]

        self.replace_threshold_frac_std = pbt_params["replace_threshold_frac_std"]
        self.replace_threshold_frac_absolute = pbt_params["replace_threshold_frac_absolute"]
        self.mutation_rate = pbt_params["mutation_rate"]
        self.change_min = pbt_params["change_min"]
        self.change_max = pbt_params["change_max"]

        self.task_name = params["task"]["name"]

        self.dbg_mode = pbt_params["dbg_mode"]

        self.policy_idx = pbt_params["policy_idx"]
        self.num_policies = pbt_params["num_policies"]

        self.num_envs = params["task"]["env"]["numEnvs"]

        self.workspace = pbt_params["workspace"]

        self.interval_steps = pbt_params["interval_steps"]
        self.start_after_steps = pbt_params["start_after"]
        self.initial_delay_steps = pbt_params["initial_delay"]

        self.params_to_mutate = pbt_params["mutation"]

        mutable_params = _flatten_params(params)
        self.mutable_params = _filter_params(mutable_params, self.params_to_mutate)

        self.with_wandb = params["wandb_activate"]


RLAlgo = Any  # just for readability


def _restart_process_with_new_params(
    policy_idx: int,
    new_params: Dict,
    restart_from_checkpoint: Optional[str],
    experiment_name: Optional[str],
    algo: Optional[RLAlgo],
    with_wandb: bool,
) -> None:
    cli_args = sys.argv

    modified_args = [cli_args[0]]  # initialize with path to the Python script
    for arg in cli_args[1:]:
        if "=" not in arg:
            modified_args.append(arg)
        else:
            assert "=" in arg
            arg_name, arg_value = arg.split("=")
            if arg_name in new_params or arg_name in [
                "checkpoint",
                "+full_experiment_name",
                "hydra.run.dir",
                "++pbt_restart",
            ]:
                # skip this parameter, it will be added later!
                continue

            modified_args.append(f"{arg_name}={arg_value}")

    modified_args.append(f"hydra.run.dir={os.getcwd()}")
    modified_args.append(f"++pbt_restart=True")

    if experiment_name is not None:
        modified_args.append(f"+full_experiment_name={experiment_name}")
    if restart_from_checkpoint is not None:
        modified_args.append(f"checkpoint={restart_from_checkpoint}")

    # add all the new (possibly mutated) parameters
    for param, value in new_params.items():
        modified_args.append(f"{param}={value}")

    if algo is not None:
        algo.writer.flush()
        algo.writer.close()

    if with_wandb:
        try:
            import wandb

            wandb.run.finish()
        except Exception as exc:
            print(f"Policy {policy_idx}: Exception {exc} in wandb.run.finish()")
            return

    print(f"Policy {policy_idx}: Restarting self with args {modified_args}", flush=True)
    os.execv(sys.executable, ["python3"] + modified_args)


def initial_pbt_check(cfg: DictConfig):
    assert cfg.pbt.enabled
    if hasattr(cfg, "pbt_restart") and cfg.pbt_restart:
        print(f"PBT job restarted from checkpoint, keep going...")
        return

    print("PBT run without 'pbt_restart=True' - must be the very start of the experiment!")
    print("Mutating initial set of hyperparameters!")

    pbt_params = PbtParams(cfg)
    new_params = mutate(
        pbt_params.mutable_params,
        pbt_params.params_to_mutate,
        pbt_params.mutation_rate,
        pbt_params.change_min,
        pbt_params.change_max,
    )
    _restart_process_with_new_params(pbt_params.policy_idx, new_params, None, None, None, False)


class PbtAlgoObserver(AlgoObserver):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.pbt_params: PbtParams = PbtParams(cfg)
        self.policy_idx: int = self.pbt_params.policy_idx
        self.num_envs: int = self.pbt_params.num_envs
        self.pbt_num_policies: int = self.pbt_params.num_policies

        self.algo: Optional[RLAlgo] = None

        self.pbt_workspace_dir = self.curr_policy_workspace_dir = None

        self.pbt_iteration = -1  # dummy value, stands for "not initialized"
        self.initial_env_frames = -1  # env frames at the beginning of the experiment, can be > 0 if we resume

        self.finished_agents = set()
        self.last_target_objectives = [_UNINITIALIZED_VALUE] * self.pbt_params.num_envs

        self.curr_target_objective_value: float = _UNINITIALIZED_VALUE
        self.target_objective_known = False  # switch to true when we have enough data to calculate target objective

        # keep track of objective values in the current iteration
        # we use best value reached in the current iteration to decide whether to be replaced by another policy
        # this reduces the noisiness of evolutionary pressure by reducing the number of situations where a policy
        # gets replaced just due to a random minor dip in performance
        self.best_objective_curr_iteration: Optional[float] = None

        self.experiment_start = time.time()

        self.with_wandb = self.pbt_params.with_wandb

    def after_init(self, algo):
        self.algo = algo

        self.pbt_workspace_dir = join(algo.train_dir, self.pbt_params.workspace)
        self.curr_policy_workspace_dir = self._policy_workspace_dir(self.pbt_params.policy_idx)
        os.makedirs(self.curr_policy_workspace_dir, exist_ok=True)

    def process_infos(self, infos, done_indices):
        if "true_objective" in infos:
            done_indices_lst = done_indices.squeeze(-1).tolist()
            self.finished_agents.update(done_indices_lst)

            for done_idx in done_indices_lst:
                true_objective_value = infos["true_objective"][done_idx].item()
                self.last_target_objectives[done_idx] = true_objective_value

            # last result for all episodes
            self.target_objective_known = len(self.finished_agents) >= self.pbt_params.num_envs
            if self.target_objective_known:
                self.curr_target_objective_value = float(np.mean(self.last_target_objectives))
        else:
            # environment does not specify "true objective", use regular reward
            # in this case, be careful not to include reward shaping coefficients into the mutation config
            self.target_objective_known = self.algo.game_rewards.current_size >= self.algo.games_to_track
            if self.target_objective_known:
                self.curr_target_objective_value = float(self.algo.mean_rewards)

        if self.target_objective_known:
            if (
                self.best_objective_curr_iteration is None
                or self.curr_target_objective_value > self.best_objective_curr_iteration
            ):
                print(
                    f"Policy {self.policy_idx}: New best objective value {self.curr_target_objective_value} in iteration {self.pbt_iteration}"
                )
                self.best_objective_curr_iteration = self.curr_target_objective_value

    def after_steps(self):
        if self.pbt_iteration == -1:
            self.pbt_iteration = self.algo.frame // self.pbt_params.interval_steps
            self.initial_env_frames = self.algo.frame
            print(
                f"Policy {self.policy_idx}: PBT init. Env frames: {self.algo.frame}, pbt_iteration: {self.pbt_iteration}"
            )

        env_frames: int = self.algo.frame
        iteration = env_frames // self.pbt_params.interval_steps
        print(
            f"Policy {self.policy_idx}: Env frames {env_frames}, iteration {iteration}, self iteration {self.pbt_iteration}"
        )

        if iteration <= self.pbt_iteration:
            return

        if not self.target_objective_known:
            # not enough data yet to calcuate avg true_objective
            print(
                f"Policy {self.policy_idx}: Not enough episodes finished, wait for more data ({len(self.finished_agents)}/{self.num_envs})..."
            )
            return
        assert self.curr_target_objective_value != _UNINITIALIZED_VALUE
        assert self.best_objective_curr_iteration is not None
        best_objective_curr_iteration: float = self.best_objective_curr_iteration

        # reset for the next iteration
        self.best_objective_curr_iteration = None
        self.target_objective_known = False

        sec_since_experiment_start = time.time() - self.experiment_start
        pbt_start_after_sec = 1 if self.pbt_params.dbg_mode else 30
        if sec_since_experiment_start < pbt_start_after_sec:
            print(
                f"Policy {self.policy_idx}: Not enough time passed since experiment start {sec_since_experiment_start}"
            )
            return

        print(f"Policy {self.policy_idx}: New pbt iteration {iteration}!")
        self.pbt_iteration = iteration

        try:
            self._save_pbt_checkpoint()
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when saving PBT checkpoint!")
            return

        try:
            checkpoints = self._load_population_checkpoints()
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when loading checkpoints!")
            return

        try:
            self._cleanup(checkpoints)
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} during cleanup!")

        policies = list(range(self.pbt_num_policies))
        target_objectives = []
        for p in policies:
            if checkpoints[p] is None:
                target_objectives.append(_UNINITIALIZED_VALUE)
            else:
                target_objectives.append(checkpoints[p]["true_objective"])

        policies_sorted = sorted(zip(target_objectives, policies), reverse=True)
        objectives = [objective for objective, p in policies_sorted]
        best_objective = objectives[0]
        policies_sorted = [p for objective, p in policies_sorted]
        best_policy = policies_sorted[0]

        self._maybe_save_best_policy(best_objective, best_policy, checkpoints[best_policy])

        objectives_filtered = [o for o in objectives if o > _UNINITIALIZED_VALUE]

        try:
            self._pbt_summaries(self.pbt_params.mutable_params, best_objective)
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when writing summaries!")
            return

        if (
            env_frames - self.initial_env_frames < self.pbt_params.start_after_steps
            or env_frames < self.pbt_params.initial_delay_steps
        ):
            print(
                f"Policy {self.policy_idx}: Not enough experience collected to replace weights. "
                f"Giving this policy more time to adjust to the latest parameters... "
                f"env_frames={env_frames} started_at={self.initial_env_frames} "
                f"restart_delay={self.pbt_params.start_after_steps} initial_delay={self.pbt_params.initial_delay_steps}"
            )
            return

        replace_worst = math.ceil(self.pbt_params.replace_fraction_worst * self.pbt_num_policies)
        replace_best = math.ceil(self.pbt_params.replace_fraction_best * self.pbt_num_policies)

        best_policies = policies_sorted[:replace_best]
        worst_policies = policies_sorted[-replace_worst:]

        print(f"Policy {self.policy_idx}: PBT best_policies={best_policies}, worst_policies={worst_policies}")

        if self.policy_idx not in worst_policies and not self.pbt_params.dbg_mode:
            # don't touch the policies that are doing okay
            print(f"Current policy {self.policy_idx} is doing well, not among the worst_policies={worst_policies}")
            return

        if best_objective_curr_iteration is not None and not self.pbt_params.dbg_mode:
            if best_objective_curr_iteration >= min(objectives[:replace_best]):
                print(
                    f"Policy {self.policy_idx}: best_objective={best_objective_curr_iteration} "
                    f"is better than some of the top policies {objectives[:replace_best]}. "
                    f"This policy should keep training for now, it is doing okay."
                )
                return

        if len(objectives_filtered) <= max(2, self.pbt_num_policies // 2) and not self.pbt_params.dbg_mode:
            print(f"Policy {self.policy_idx}: Not enough data to start PBT, {objectives_filtered}")
            return

        print(f"Current policy {self.policy_idx} is among the worst_policies={worst_policies}, consider replacing weights")
        print(
            f"Policy {self.policy_idx} objective: {self.curr_target_objective_value}, best_objective={best_objective} (best_policy={best_policy})."
        )

        replacement_policy_candidate = random.choice(best_policies)
        candidate_objective = checkpoints[replacement_policy_candidate]["true_objective"]
        targ_objective_value = self.curr_target_objective_value
        objective_delta = candidate_objective - targ_objective_value

        num_outliers = int(math.floor(0.2 * len(objectives_filtered)))
        print(f"Policy {self.policy_idx} num outliers: {num_outliers}")

        if len(objectives_filtered) > num_outliers:
            objectives_filtered_sorted = sorted(objectives_filtered)

            # remove the worst policies from the std calculation, this will allow us to keep improving even if 1-2 policies
            # crashed and can't keep improving. Otherwise, std value will be too large.
            objectives_std = np.std(objectives_filtered_sorted[num_outliers:])
        else:
            objectives_std = np.std(objectives_filtered)

        objective_threshold = self.pbt_params.replace_threshold_frac_std * objectives_std

        absolute_threshold = self.pbt_params.replace_threshold_frac_absolute * abs(candidate_objective)

        if objective_delta > objective_threshold and objective_delta > absolute_threshold:
            # replace this policy with a candidate
            replacement_policy = replacement_policy_candidate
            print(f"Replacing underperforming policy {self.policy_idx} with {replacement_policy}")
        else:
            print(
                f"Policy {self.policy_idx}: Difference in objective value ({candidate_objective} vs {targ_objective_value}) is not sufficient to justify replacement,"
                f"{objective_delta}, {objectives_std}, {objective_threshold}, {absolute_threshold}"
            )

            # replacing with "self": keep the weights but mutate the hyperparameters
            replacement_policy = self.policy_idx

        # Decided to replace the policy weights!

        # we can either copy parameters from the checkpoint we're restarting from, or keep our parameters and
        # further mutate them.
        if random.random() < 0.5:
            new_params = checkpoints[replacement_policy]["params"]
        else:
            new_params = self.pbt_params.mutable_params

        new_params = mutate(
            new_params,
            self.pbt_params.params_to_mutate,
            self.pbt_params.mutation_rate,
            self.pbt_params.change_min,
            self.pbt_params.change_max,
        )

        experiment_name = checkpoints[self.policy_idx]["experiment_name"]

        try:
            self._pbt_summaries(new_params, best_objective)
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when writing summaries!")
            return

        try:
            restart_checkpoint = os.path.abspath(checkpoints[replacement_policy]["checkpoint"])

            # delete previous tempdir to make sure we don't grow too big
            checkpoint_tmp_dir = join(project_tmp_dir(), f"{experiment_name}_p{self.policy_idx}")
            if os.path.isdir(checkpoint_tmp_dir):
                shutil.rmtree(checkpoint_tmp_dir)

            checkpoint_tmp_dir = safe_ensure_dir_exists(checkpoint_tmp_dir)
            restart_checkpoint_tmp = join(checkpoint_tmp_dir, os.path.basename(restart_checkpoint))

            # copy the checkpoint file to the temp dir to make sure it does not get deleted while we're restarting
            shutil.copyfile(restart_checkpoint, restart_checkpoint_tmp)
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when copying checkpoint file for restart")
            # perhaps checkpoint file was deleted before we could make a copy. Abort the restart.
            return

        # try to load the checkpoint file and if it fails, abandon the restart
        try:
            self._rewrite_checkpoint(restart_checkpoint_tmp, env_frames)
        except Exception as exc:
            # this should happen infrequently so should not affect training in any significant way
            print(
                f"Policy {self.policy_idx}: Exception {exc} when loading checkpoint file for restart."
                f"Aborting restart. Continue training with the existing set of weights!"
            )
            return

        print(
            f"Policy {self.policy_idx}: Preparing to restart the process with mutated parameters! "
            f"Checkpoint {restart_checkpoint_tmp}"
        )
        _restart_process_with_new_params(
            self.policy_idx, new_params, restart_checkpoint_tmp, experiment_name, self.algo, self.with_wandb
        )

    def _rewrite_checkpoint(self, restart_checkpoint_tmp: str, env_frames: int) -> None:
        state = torch.load(restart_checkpoint_tmp)
        print(f"Policy {self.policy_idx}: restarting from checkpoint {restart_checkpoint_tmp}, {state['frame']}")
        print(f"Replacing {state['frame']} with {env_frames}...")
        state["frame"] = env_frames

        pbt_history = state.get("pbt_history", [])
        print(f"PBT history: {pbt_history}")
        pbt_history.append((self.policy_idx, env_frames, self.curr_target_objective_value))
        state["pbt_history"] = pbt_history

        torch.save(state, restart_checkpoint_tmp)
        print(f"Policy {self.policy_idx}: checkpoint rewritten to {restart_checkpoint_tmp}!")

    def _save_pbt_checkpoint(self):
        """Save PBT-specific information including iteration number, policy index and hyperparameters."""
        checkpoint_file = join(self.curr_policy_workspace_dir, _model_checkpnt_name(self.pbt_iteration))
        algo_state = self.algo.get_full_state_weights()
        safe_save(algo_state, checkpoint_file)

        pbt_checkpoint_file = join(self.curr_policy_workspace_dir, _checkpnt_name(self.pbt_iteration))

        pbt_checkpoint = {
            "iteration": self.pbt_iteration,
            "true_objective": self.curr_target_objective_value,
            "frame": self.algo.frame,
            "params": self.pbt_params.mutable_params,
            "checkpoint": os.path.abspath(checkpoint_file),
            "pbt_checkpoint": os.path.abspath(pbt_checkpoint_file),
            "experiment_name": self.algo.experiment_name,
        }

        with open(pbt_checkpoint_file, "w") as fobj:
            print(f"Policy {self.policy_idx}: Saving {pbt_checkpoint_file}...")
            yaml.dump(pbt_checkpoint, fobj)

    def _policy_workspace_dir(self, policy_idx):
        return join(self.pbt_workspace_dir, f"{policy_idx:03d}")

    def _load_population_checkpoints(self):
        """
        Load checkpoints for other policies in the population.
        Pick the newest checkpoint, but not newer than our current iteration.
        """
        checkpoints = dict()

        for policy_idx in range(self.pbt_num_policies):
            checkpoints[policy_idx] = None

            policy_workspace_dir = self._policy_workspace_dir(policy_idx)

            if not os.path.isdir(policy_workspace_dir):
                continue

            pbt_checkpoint_files = [f for f in os.listdir(policy_workspace_dir) if f.endswith(".yaml")]
            pbt_checkpoint_files.sort(reverse=True)

            for pbt_checkpoint_file in pbt_checkpoint_files:
                iteration_str = pbt_checkpoint_file.split(".")[0]
                iteration = int(iteration_str)

                if iteration <= self.pbt_iteration:
                    with open(join(policy_workspace_dir, pbt_checkpoint_file), "r") as fobj:
                        print(f"Policy {self.policy_idx}: Loading policy-{policy_idx} {pbt_checkpoint_file}")
                        checkpoints[policy_idx] = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                        break
                else:
                    # print(f'Policy {self.policy_idx}: Ignoring {pbt_checkpoint_file} because it is newer than our current iteration')
                    pass

        assert self.policy_idx in checkpoints.keys()
        return checkpoints

    def _maybe_save_best_policy(self, best_objective, best_policy_idx: int, best_policy_checkpoint):
        # make a directory containing the best policy checkpoints using safe_filesystem_op
        best_policy_workspace_dir = join(self.pbt_workspace_dir, f"best{self.policy_idx}")
        safe_filesystem_op(os.makedirs, best_policy_workspace_dir, exist_ok=True)

        best_objective_so_far = _UNINITIALIZED_VALUE

        best_policy_checkpoint_files = [f for f in os.listdir(best_policy_workspace_dir) if f.endswith(".yaml")]
        best_policy_checkpoint_files.sort(reverse=True)
        if best_policy_checkpoint_files:
            with open(join(best_policy_workspace_dir, best_policy_checkpoint_files[0]), "r") as fobj:
                best_policy_checkpoint_so_far = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                best_objective_so_far = best_policy_checkpoint_so_far["true_objective"]

        if best_objective_so_far >= best_objective:
            # don't save the checkpoint if it is worse than the best checkpoint so far
            return

        print(f"Policy {self.policy_idx}: New best objective: {best_objective}!")

        # save the best policy checkpoint to this folder
        best_policy_checkpoint_name = f"{self.pbt_params.task_name}_best_obj_{best_objective:015.5f}_iter_{self.pbt_iteration:04d}_policy{best_policy_idx:03d}_frame{self.algo.frame}"

        # copy the checkpoint file to the best policy directory
        try:
            shutil.copy(
                best_policy_checkpoint["checkpoint"],
                join(best_policy_workspace_dir, f"{best_policy_checkpoint_name}.pth"),
            )
            shutil.copy(
                best_policy_checkpoint["pbt_checkpoint"],
                join(best_policy_workspace_dir, f"{best_policy_checkpoint_name}.yaml"),
            )

            # cleanup older best policy checkpoints, we want to keep only N latest files
            best_policy_checkpoint_files = [f for f in os.listdir(best_policy_workspace_dir)]
            best_policy_checkpoint_files.sort(reverse=True)

            n_to_keep = 6
            for best_policy_checkpoint_file in best_policy_checkpoint_files[n_to_keep:]:
                os.remove(join(best_policy_workspace_dir, best_policy_checkpoint_file))

        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when copying best checkpoint!")
            # no big deal if this fails, hopefully the next time we will succeeed
            return

    def _pbt_summaries(self, params, best_objective):
        for param, value in params.items():
            self.algo.writer.add_scalar(f"pbt/{param}", value, self.algo.frame)
        self.algo.writer.add_scalar(f"pbt/00_best_objective", best_objective, self.algo.frame)
        self.algo.writer.flush()

    def _cleanup(self, checkpoints):
        iterations = []
        for policy_idx, checkpoint in checkpoints.items():
            if checkpoint is None:
                iterations.append(0)
            else:
                iterations.append(checkpoint["iteration"])

        oldest_iteration = sorted(iterations)[0]
        cleanup_threshold = oldest_iteration - 20
        print(
            f"Policy {self.policy_idx}: Oldest iteration in population is {oldest_iteration}, removing checkpoints older than {cleanup_threshold} iteration"
        )

        pbt_checkpoint_files = [f for f in os.listdir(self.curr_policy_workspace_dir)]

        for f in pbt_checkpoint_files:
            if "." in f:
                iteration_idx = int(f.split(".")[0])
                if iteration_idx <= cleanup_threshold:
                    print(f"Policy {self.policy_idx}: PBT cleanup: removing checkpoint {f}")
                    # we catch all exceptions in this function so no need to use safe_filesystem_op
                    os.remove(join(self.curr_policy_workspace_dir, f))

        # Sometimes, one of the PBT processes can get stuck, or crash, or be scheduled significantly later on Slurm
        # or a similar cluster management system.
        # In that case, we will accumulate a lot of older checkpoints. In order to keep the number of older checkpoints
        # under control (to avoid running out of disk space) we implement the following logic:
        # when we have more than N checkpoints, we delete half of the oldest checkpoints. This caps the max amount of
        # disk space used, and still allows older policies to participate in PBT

        max_old_checkpoints = 25
        while True:
            pbt_checkpoint_files = [f for f in os.listdir(self.curr_policy_workspace_dir) if f.endswith(".yaml")]
            if len(pbt_checkpoint_files) <= max_old_checkpoints:
                break
            if not self._delete_old_checkpoint(pbt_checkpoint_files):
                break

    def _delete_old_checkpoint(self, pbt_checkpoint_files: List[str]) -> bool:
        """
        Delete the checkpoint that results in the smallest max gap between the remaining checkpoints.
        Do not delete any of the last N checkpoints.
        """
        pbt_checkpoint_files.sort()
        n_latest_to_keep = 10
        candidates = pbt_checkpoint_files[:-n_latest_to_keep]
        num_candidates = len(candidates)
        if num_candidates < 3:
            return False

        def _iter(f):
            return int(f.split(".")[0])

        best_gap = 1e9
        best_candidate = 1
        for i in range(1, num_candidates - 1):
            prev_iteration = _iter(candidates[i - 1])
            next_iteration = _iter(candidates[i + 1])

            # gap is we delete the ith candidate
            gap = next_iteration - prev_iteration
            if gap < best_gap:
                best_gap = gap
                best_candidate = i

        # delete the best candidate
        best_candidate_file = candidates[best_candidate]
        files_to_remove = [best_candidate_file, _model_checkpnt_name(_iter(best_candidate_file))]
        for file_to_remove in files_to_remove:
            print(
                f"Policy {self.policy_idx}: PBT cleanup old checkpoints, removing checkpoint {file_to_remove} (best gap {best_gap})"
            )
            os.remove(join(self.curr_policy_workspace_dir, file_to_remove))

        return True
