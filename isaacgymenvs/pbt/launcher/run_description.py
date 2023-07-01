import os
import re
from collections import OrderedDict
from os.path import join

import numpy as np


class ParamGenerator:
    def __init__(self):
        pass

    def generate_params(self, randomize=True):
        """Supposed to be a generator (so should yield dicts of parameters)."""
        pass


class ParamList(ParamGenerator):
    """The most simple kind of generator, represents just the list of parameter combinations."""

    def __init__(self, combinations):
        super(ParamList, self).__init__()
        self.combinations = combinations

    def generate_params(self, randomize=True):
        if randomize:
            combinations = np.random.permutation(self.combinations)
        else:
            combinations = self.combinations

        for combination in combinations:
            yield combination


class ParamGrid(ParamGenerator):
    """Parameter generator for grid search."""

    def __init__(self, grid_tuples):
        """Uses OrderedDict, so must be initialized with the list of tuples if you want to preserve order."""
        super(ParamGrid, self).__init__()
        self.grid = OrderedDict(grid_tuples)

    def _generate_combinations(self, param_idx, params):
        """Recursively generate all parameter combinations in a grid."""

        if param_idx == len(self.grid) - 1:
            # last parameter, just return list of values for this parameter
            return [[value] for value in self.grid[params[param_idx]]]
        else:
            subcombinations = self._generate_combinations(param_idx + 1, params)  # returns list of param combinations
            result = []

            # iterate over all values of current parameter
            for value in self.grid[params[param_idx]]:
                for subcombination in subcombinations:
                    result.append([value] + subcombination)

            return result

    def generate_params(self, randomize=False):
        if len(self.grid) == 0:
            return dict()

        # start with 0th value for every parameter
        total_num_combinations = np.prod([len(p_values) for p_values in self.grid.values()])

        param_names = tuple(self.grid.keys())
        all_combinations = self._generate_combinations(0, param_names)

        assert len(all_combinations) == total_num_combinations

        if randomize:
            all_combinations = np.random.permutation(all_combinations)

        for combination in all_combinations:
            combination_dict = dict()
            for i, param_name in enumerate(param_names):
                if isinstance(param_name, (list, tuple)):
                    for j, param in enumerate(param_name):
                        combination_dict[param] = combination[i][j]
                else:
                    combination_dict[param_name] = combination[i]

            yield combination_dict


class Experiment:
    def __init__(self, name, cmd, param_generator=(), env_vars=None):
        """
        :param cmd: base command to append the parameters to
        :param param_generator: iterable of parameter dicts
        """
        self.base_name = name
        self.cmd = cmd
        self.params = list(param_generator)
        self.env_vars = env_vars

    def generate_experiments(self, experiment_arg_name, customize_experiment_name, param_prefix):
        """Yields tuples of (cmd, experiment_name)"""
        num_experiments = 1 if len(self.params) == 0 else len(self.params)

        for experiment_idx in range(num_experiments):
            cmd_tokens = [self.cmd]
            experiment_name_tokens = [self.base_name]

            # abbreviations for parameter names that we've used
            param_shorthands = []

            if len(self.params) > 0:
                params = self.params[experiment_idx]
                for param, value in params.items():
                    param_str = f"{param_prefix}{param}={value}"
                    cmd_tokens.append(param_str)

                    param_tokens = re.split("[._-]", param)
                    shorthand_tokens = [t[0] for t in param_tokens[:-1]]

                    last_token_l = min(3, len(param_tokens[-1]))
                    shorthand = ".".join(shorthand_tokens + [param_tokens[-1][:last_token_l]])
                    while last_token_l <= len(param_tokens[-1]) and shorthand in param_shorthands:
                        last_token_l += 1
                        shorthand = ".".join(shorthand_tokens + [param_tokens[-1][:last_token_l]])

                    param_shorthands.append(shorthand)
                    experiment_name_token = f"{shorthand}_{value}"
                    experiment_name_tokens.append(experiment_name_token)

            if customize_experiment_name:
                experiment_name = f"{experiment_idx:02d}_" + "_".join(experiment_name_tokens)
                if len(experiment_name) > 100:
                    print(f"Experiment name is extra long! ({len(experiment_name)} characters)")
            else:
                experiment_name = f"{experiment_idx:02d}_{self.base_name}"

            cmd_tokens.append(f"{experiment_arg_name}={experiment_name}")
            param_str = " ".join(cmd_tokens)

            yield param_str, experiment_name


class RunDescription:
    def __init__(
        self,
        run_name,
        experiments,
        experiment_arg_name="--experiment",
        experiment_dir_arg_name="--train_dir",
        customize_experiment_name=True,
        param_prefix="--",
    ):
        """
        :param run_name: overall name of the experiment and the name of the root folder
        :param experiments: a list of Experiment objects to run
        :param experiment_arg_name: CLI argument of the underlying experiment that determines it's unique name
               to be generated by the launcher. Default: --experiment
        :param experiment_dir_arg_name: CLI argument for the root train dir of your experiment. Default: --train_dir
        :param customize_experiment_name: whether to add a hyperparameter combination to the experiment name
        :param param_prefix: most experiments will use "--" prefix for each parameter, but some apps don't have this
               prefix, i.e. with Hydra you should set it to empty string.
        """
        self.run_name = run_name
        self.experiments = experiments
        self.experiment_suffix = ""

        self.experiment_arg_name = experiment_arg_name
        self.experiment_dir_arg_name = experiment_dir_arg_name

        self.customize_experiment_name = customize_experiment_name

        self.param_prefix = param_prefix

    def generate_experiments(self, train_dir, makedirs=True):
        """Yields tuples (final cmd for experiment, experiment_name, root_dir)."""
        for experiment in self.experiments:
            root_dir = join(self.run_name, f"{experiment.base_name}_{self.experiment_suffix}")

            experiment_cmds = experiment.generate_experiments(
                self.experiment_arg_name, self.customize_experiment_name, self.param_prefix
            )
            for experiment_cmd, experiment_name in experiment_cmds:
                experiment_dir = join(train_dir, root_dir)
                if makedirs:
                    os.makedirs(experiment_dir, exist_ok=True)
                experiment_cmd += f" {self.experiment_dir_arg_name}={experiment_dir}"
                yield experiment_cmd, experiment_name, root_dir, experiment.env_vars
